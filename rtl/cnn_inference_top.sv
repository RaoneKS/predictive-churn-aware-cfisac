// CNN inference datapath using the validated 8x8 systolic adapter.
// Conv1 uses one K=8 tile (5 logical inputs plus zero padding).
// Conv2 uses three K=8 tiles; INT32 partial sums are combined before bias/requantization.
// GAP remains a reduction operation; the classifier uses one K=8 systolic tile.

`timescale 1ns/1ps
import pkg_accelerator::*;

module cnn_inference_top (
  input logic clk,
  input logic rst_n,
  input logic start,
  output logic done,
  input logic signed [7:0] window_sample,
  input logic window_valid,
  output logic window_ready,
  input logic signed [319:0] conv1_weights_flat,
  input logic signed [255:0] conv1_bias_flat,
  input logic signed [1535:0] conv2_weights_flat,
  input logic signed [255:0] conv2_bias_flat,
  input logic signed [255:0] classifier_weights_flat,
  input logic signed [127:0] classifier_bias_flat,
  output logic signed [31:0] logits0,
  output logic signed [31:0] logits1,
  output logic signed [31:0] logits2,
  output logic signed [31:0] logits3,
  output logic logits_valid,
  output logic [31:0] cycle_count,
  output logic overflow_seen
);
  localparam integer W=256, P=128, C=8;

  typedef enum logic [3:0] {
    IDLE, INPUT, CONV1_SETUP, CONV1_WAIT, POOL,
    CONV2_SETUP, CONV2_WAIT, GAP,
    CLASSIFIER_SETUP, CLASSIFIER_WAIT, FINISH
  } state_t;
  state_t state;

  logic signed [7:0] x[0:255];
  logic signed [7:0] c1[0:255][0:7];
  logic signed [7:0] pool[0:127][0:7];
  logic signed [7:0] c2[0:127][0:7];
  logic signed [7:0] gap[0:7];

  logic tile_start, tile_busy, tile_done, tile_overflow;
  logic [ACT_WIDTH*ARRAY_ROWS-1:0] tile_act;
  logic [WGT_WIDTH*ARRAY_ROWS-1:0] tile_weights[0:ARRAY_ROWS-1];
  logic [ACC_WIDTH*ARRAY_COLS-1:0] tile_acc;
  logic signed [31:0] partial_sum[0:7];

  integer idx, tile_idx;
  integer oc, k, ch;
  integer input_index, flat_k, src_ch, kernel_pos;
  integer acc_tmp[0:7], sum_tmp[0:7];
  logic signed [31:0] gap_acc[0:7];
  integer q_tmp, log_tmp, map_value;

  // Constant 24-entry mapping replaces runtime divide/modulo-by-3 logic.
  // flat_k is in [0,23], ordered as [input_channel][kernel_position].
  function automatic logic [4:0] conv2_k_map(input integer flat_index);
    begin
      case (flat_index)
        0:  conv2_k_map = {3'd0,2'd0};
        1:  conv2_k_map = {3'd0,2'd1};
        2:  conv2_k_map = {3'd0,2'd2};
        3:  conv2_k_map = {3'd1,2'd0};
        4:  conv2_k_map = {3'd1,2'd1};
        5:  conv2_k_map = {3'd1,2'd2};
        6:  conv2_k_map = {3'd2,2'd0};
        7:  conv2_k_map = {3'd2,2'd1};
        8:  conv2_k_map = {3'd2,2'd2};
        9:  conv2_k_map = {3'd3,2'd0};
        10: conv2_k_map = {3'd3,2'd1};
        11: conv2_k_map = {3'd3,2'd2};
        12: conv2_k_map = {3'd4,2'd0};
        13: conv2_k_map = {3'd4,2'd1};
        14: conv2_k_map = {3'd4,2'd2};
        15: conv2_k_map = {3'd5,2'd0};
        16: conv2_k_map = {3'd5,2'd1};
        17: conv2_k_map = {3'd5,2'd2};
        18: conv2_k_map = {3'd6,2'd0};
        19: conv2_k_map = {3'd6,2'd1};
        20: conv2_k_map = {3'd6,2'd2};
        21: conv2_k_map = {3'd7,2'd0};
        22: conv2_k_map = {3'd7,2'd1};
        23: conv2_k_map = {3'd7,2'd2};
        default: conv2_k_map = {3'd0,2'd0};
      endcase
    end
  endfunction

  assign window_ready = (state == INPUT);
  assign done = (state == FINISH);
  assign logits_valid = (state == FINISH);

  cnn_systolic_adapter u_tile (
    .clk(clk),
    .rst_n(rst_n),
    .start(tile_start),
    .act_vector(tile_act),
    .weight_rows(tile_weights),
    .busy(tile_busy),
    .done(tile_done),
    .acc_out_packed(tile_acc),
    .any_overflow(tile_overflow)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      idx <= 0;
      tile_idx <= 0;
      cycle_count <= 0;
      overflow_seen <= 1'b0;
      tile_start <= 1'b0;
      tile_act <= '0;
      logits0 <= 0;
      logits1 <= 0;
      logits2 <= 0;
      logits3 <= 0;
      for (k=0; k<ARRAY_ROWS; k=k+1)
        tile_weights[k] <= '0;
      for (oc=0; oc<C; oc=oc+1) begin
        gap[oc] <= 0;
        partial_sum[oc] <= 0;
        gap_acc[oc] <= 0;
      end
    end else begin
      cycle_count <= cycle_count + 1'b1;
      // Default to a one-cycle start pulse; setup states assert it.
      tile_start <= 1'b0;
      if (tile_done && tile_overflow)
        overflow_seen <= 1'b1;

      case (state)
        IDLE: begin
          idx <= 0;
          tile_idx <= 0;
          if (start)
            state <= INPUT;
        end

        INPUT: begin
          if (window_valid && window_ready) begin
            x[idx] <= window_sample;
            if (idx == W-1) begin
              idx <= 0;
              state <= CONV1_SETUP;
            end else begin
              idx <= idx + 1;
            end
          end
        end

        // Conv1: K=5 logical features, padded to one K=8 systolic tile.
        CONV1_SETUP: begin
          tile_act <= '0;
          for (k=0; k<ARRAY_ROWS; k=k+1) begin
            tile_weights[k] <= '0;
            if (k < 5) begin
              input_index = idx + k - 2;
              if (input_index >= 0 && input_index < W)
                tile_act[k*ACT_WIDTH +: ACT_WIDTH] <= x[input_index];
              for (oc=0; oc<C; oc=oc+1)
                tile_weights[k][oc*WGT_WIDTH +: WGT_WIDTH] <=
                  conv1_weights_flat[(k*C+oc)*WGT_WIDTH +: WGT_WIDTH];
            end
          end
          tile_start <= 1'b1;
          state <= CONV1_WAIT;
        end

        CONV1_WAIT: begin
          if (tile_done) begin
            for (oc=0; oc<C; oc=oc+1) begin
              acc_tmp[oc] = $signed(tile_acc[oc*ACC_WIDTH +: ACC_WIDTH])
                          + $signed(conv1_bias_flat[oc*32 +: 32]);
              if (acc_tmp[oc] < 0)
                acc_tmp[oc] = 0;
              q_tmp = acc_tmp[oc] >>> 8;
              if (q_tmp > 127) q_tmp = 127;
              if (q_tmp < 0) q_tmp = 0;
              c1[idx][oc] <= q_tmp[7:0];
            end
            if (idx == W-1) begin
              idx <= 0;
              state <= POOL;
            end else begin
              idx <= idx + 1;
              state <= CONV1_SETUP;
            end
          end
        end

        POOL: begin
          for (ch=0; ch<C; ch=ch+1)
            pool[idx][ch] <= (c1[2*idx][ch] > c1[2*idx+1][ch])
                           ? c1[2*idx][ch] : c1[2*idx+1][ch];
          if (idx == P-1) begin
            idx <= 0;
            tile_idx <= 0;
            state <= CONV2_SETUP;
          end else begin
            idx <= idx + 1;
          end
        end

        // Conv2: K=24 flattened as [input_channel, kernel_position].
        // Process three K=8 tiles and preserve INT32 partial sums between tiles.
        CONV2_SETUP: begin
          tile_act <= '0;
          for (k=0; k<ARRAY_ROWS; k=k+1) begin
            flat_k = tile_idx*ARRAY_ROWS + k;
            map_value = conv2_k_map(flat_k);
            src_ch = (map_value >> 2) & 7;
            kernel_pos = map_value & 3;
            input_index = idx + kernel_pos - 1;
            tile_weights[k] <= '0;
            if (flat_k < 24) begin
              if (input_index >= 0 && input_index < P)
                tile_act[k*ACT_WIDTH +: ACT_WIDTH] <= pool[input_index][src_ch];
              for (oc=0; oc<C; oc=oc+1)
                tile_weights[k][oc*WGT_WIDTH +: WGT_WIDTH] <=
                  conv2_weights_flat[(flat_k*C+oc)*WGT_WIDTH +: WGT_WIDTH];
            end
          end
          tile_start <= 1'b1;
          state <= CONV2_WAIT;
        end

        CONV2_WAIT: begin
          if (tile_done) begin
            if (tile_idx < 2) begin
              for (oc=0; oc<C; oc=oc+1)
                partial_sum[oc] <= partial_sum[oc]
                                  + $signed(tile_acc[oc*ACC_WIDTH +: ACC_WIDTH]);
              tile_idx <= tile_idx + 1;
              state <= CONV2_SETUP;
            end else begin
              for (oc=0; oc<C; oc=oc+1) begin
                acc_tmp[oc] = partial_sum[oc]
                            + $signed(tile_acc[oc*ACC_WIDTH +: ACC_WIDTH])
                            + $signed(conv2_bias_flat[oc*32 +: 32]);
                if (acc_tmp[oc] < 0)
                  acc_tmp[oc] = 0;
                q_tmp = acc_tmp[oc] >>> 8;
                if (q_tmp > 127) q_tmp = 127;
                if (q_tmp < 0) q_tmp = 0;
                c2[idx][oc] <= q_tmp[7:0];
                partial_sum[oc] <= 0;
              end
              tile_idx <= 0;
              if (idx == P-1) begin
                idx <= 0;
                for (ch=0; ch<C; ch=ch+1)
                  gap_acc[ch] <= 0;
                state <= GAP;
              end else begin
                idx <= idx + 1;
                state <= CONV2_SETUP;
              end
            end
          end
        end

        // Sequential GAP: accumulate one spatial position per cycle.
        // This replaces an unrolled 128-term combinational adder chain per
        // channel with eight 32-bit accumulators and eight adders.
        GAP: begin
          for (ch=0; ch<C; ch=ch+1) begin
            sum_tmp[ch] = $signed(gap_acc[ch]) + $signed(c2[idx][ch]);
            gap_acc[ch] <= sum_tmp[ch];
            if (idx == P-1) begin
              // Include the current (last) sample before applying /128.
              // Conv2 has ReLU, so the GAP sum is nonnegative.
              q_tmp = sum_tmp[ch] >>> 7;
              if (q_tmp > 127) q_tmp = 127;
              if (q_tmp < -128) q_tmp = -128;
              gap[ch] <= q_tmp[7:0];
            end
          end
          if (idx == P-1) begin
            idx <= 0;
            state <= CLASSIFIER_SETUP;
          end else begin
            idx <= idx + 1;
          end
        end

        // Dense classifier: K=8, four logical output columns.
        CLASSIFIER_SETUP: begin
          for (k=0; k<ARRAY_ROWS; k=k+1) begin
            tile_act[k*ACT_WIDTH +: ACT_WIDTH] <= gap[k];
            tile_weights[k] <= '0;
            for (oc=0; oc<4; oc=oc+1)
              tile_weights[k][oc*WGT_WIDTH +: WGT_WIDTH] <=
                classifier_weights_flat[(k*4+oc)*WGT_WIDTH +: WGT_WIDTH];
          end
          tile_start <= 1'b1;
          state <= CLASSIFIER_WAIT;
        end

        CLASSIFIER_WAIT: begin
          if (tile_done) begin
            log_tmp = $signed(tile_acc[0*ACC_WIDTH +: ACC_WIDTH])
                    + $signed(classifier_bias_flat[0*32 +: 32]);
            q_tmp = log_tmp >>> 5;
            if (q_tmp > 127) q_tmp = 127;
            if (q_tmp < -128) q_tmp = -128;
            logits0 <= q_tmp;

            log_tmp = $signed(tile_acc[1*ACC_WIDTH +: ACC_WIDTH])
                    + $signed(classifier_bias_flat[1*32 +: 32]);
            q_tmp = log_tmp >>> 5;
            if (q_tmp > 127) q_tmp = 127;
            if (q_tmp < -128) q_tmp = -128;
            logits1 <= q_tmp;

            log_tmp = $signed(tile_acc[2*ACC_WIDTH +: ACC_WIDTH])
                    + $signed(classifier_bias_flat[2*32 +: 32]);
            q_tmp = log_tmp >>> 5;
            if (q_tmp > 127) q_tmp = 127;
            if (q_tmp < -128) q_tmp = -128;
            logits2 <= q_tmp;

            log_tmp = $signed(tile_acc[3*ACC_WIDTH +: ACC_WIDTH])
                    + $signed(classifier_bias_flat[3*32 +: 32]);
            q_tmp = log_tmp >>> 5;
            if (q_tmp > 127) q_tmp = 127;
            if (q_tmp < -128) q_tmp = -128;
            logits3 <= q_tmp;
            state <= FINISH;
          end
        end

        FINISH: state <= IDLE;
        default: state <= IDLE;
      endcase
    end
  end
endmodule
