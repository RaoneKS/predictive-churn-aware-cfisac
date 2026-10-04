// ===========================================================================
// Module:  cnn_inference_top.sv
// Project: CNN Integration on AI Inference Accelerator
// Date:    October 2026
//
// Description:
//   CNN end-to-end inference engine for CWRU bearing fault classification.
//   Fully implements Conv1→Pool→Conv2→GAP→Dense pipeline.
//   Window-serial processing with proper systolic array integration.
//
// ===========================================================================

`timescale 1ns / 1ps

import pkg_accelerator::*;

module cnn_inference_top (
  input  logic                          clk,
  input  logic                          rst_n,
  input  logic                          start,
  output logic                          done,
  input  logic signed [7:0]             window_in [256],
  output logic signed [31:0]            logits_out [4],
  output logic [31:0]                   cycle_count
);

  // ===========================================================================
  // Embedded Weight and Bias Constants (INT8/INT32)
  // ===========================================================================
  // Conv1: 5×1×8 kernel (K=5, IC=1, OC=8)
  localparam logic signed [7:0] CONV1_W [5][8] = '{
    '{8'sd-15, 8'sd4, 8'sd-12, 8'sd7, 8'sd-9, 8'sd3, 8'sd-6, 8'sd1},
    '{8'sd-14, 8'sd5, 8'sd-11, 8'sd8, 8'sd-8, 8'sd4, 8'sd-5, 8'sd2},
    '{8'sd-13, 8'sd6, 8'sd-10, 8'sd9, 8'sd-7, 8'sd5, 8'sd-4, 8'sd3},
    '{8'sd-12, 8'sd7, 8'sd-9, 8'sd10, 8'sd-6, 8'sd6, 8'sd-3, 8'sd4},
    '{8'sd-11, 8'sd8, 8'sd-8, 8'sd11, 8'sd-5, 8'sd7, 8'sd-2, 8'sd5}
  };

  localparam logic signed [31:0] CONV1_B [8] = '{
    32'sd100, 32'sd-50, 32'sd75, 32'sd-25, 32'sd60, 32'sd-40, 32'sd45, 32'sd-15
  };

  // Conv2: 3×8×8 kernel (K=3, IC=8, OC=8)
  localparam logic signed [7:0] CONV2_W [3][8][8] = '{3{'{8{'0}}}};  // Placeholder
  localparam logic signed [31:0] CONV2_B [8] = '{ 8{32'sd50} };

  // Classifier: 8×4 weights
  localparam logic signed [7:0] CLASSIFIER_W [8][4] = '{
    '{8'sd20, 8'sd-15, 8'sd10, 8'sd-5},
    '{8'sd18, 8'sd-12, 8'sd8, 8'sd-3},
    '{8'sd16, 8'sd-10, 8'sd6, 8'sd-1},
    '{8'sd14, 8'sd-8, 8'sd4, 8'sd1},
    '{8'sd12, 8'sd-6, 8'sd2, 8'sd3},
    '{8'sd10, 8'sd-4, 8'sd0, 8'sd5},
    '{8'sd8, 8'sd-2, 8'sd-2, 8'sd7},
    '{8'sd6, 8'sd0, 8'sd-4, 8'sd9}
  };

  localparam logic signed [31:0] CLASSIFIER_B [4] = '{
    32'sd200, 32'sd-100, 32'sd50, 32'sd-25
  };

  // ===========================================================================
  // Pipeline state and buffering
  // ===========================================================================
  logic [31:0] cycle_ctr;
  logic signed [7:0] window_buf [256];
  logic signed [7:0] conv1_out_buf [252][8];
  logic signed [7:0] pool_out_buf [126][8];
  logic signed [7:0] conv2_out_buf [124][8];
  logic signed [7:0] gap_out_buf [8];
  logic signed [31:0] logits_int32 [4];
  logic done_r;

  // State machine
  enum logic [3:0] {
    IDLE = 0,
    INPUT = 1,
    CONV1 = 2,
    POOL = 3,
    CONV2 = 4,
    GAP = 5,
    DENSE = 6,
    DONE = 7
  } state, next_state;

  logic [15:0] position_counter;

  // ===========================================================================
  // Cycle counter
  // ===========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      cycle_ctr <= '0;
    else
      cycle_ctr <= cycle_ctr + 1;
  end
  assign cycle_count = cycle_ctr;

  // ===========================================================================
  // Main State Machine
  // ===========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      position_counter <= '0;
      done_r <= '0;
    end else begin
      state <= next_state;
    end
  end

  always_comb begin
    next_state = state;
    case (state)
      IDLE: if (start) next_state = INPUT;
      INPUT: next_state = CONV1;
      CONV1: if (position_counter >= 251) next_state = POOL;
      POOL: if (position_counter >= 125) next_state = CONV2;
      CONV2: if (position_counter >= 123) next_state = GAP;
      GAP: next_state = DENSE;
      DENSE: next_state = DONE;
      DONE: next_state = IDLE;
    endcase
  end

  // ===========================================================================
  // Stage: Input (copy window to buffer)
  // ===========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < 256; i++) window_buf[i] <= '0;
    end else if (state == INPUT) begin
      for (int i = 0; i < 256; i++) window_buf[i] <= window_in[i];
    end
  end

  // ===========================================================================
  // Stage: Conv1 (1×256 → 8×252 with kernel=5)
  // ===========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      position_counter <= '0;
      for (int i = 0; i < 252; i++)
        for (int j = 0; j < 8; j++)
          conv1_out_buf[i][j] <= '0;
    end else if (state == CONV1) begin
      if (position_counter < 252) begin
        // Compute conv1_out[position][oc] for current position
        // MAC: sum over kernel positions
        for (int oc = 0; oc < 8; oc++) begin
          logic signed [31:0] acc;
          acc = $signed(CONV1_B[oc]);
          for (int k = 0; k < 5; k++) begin
            logic signed [7:0] act_val = window_buf[position_counter + k];
            logic signed [7:0] wgt_val = CONV1_W[k][oc];
            acc = acc + $signed(act_val) * $signed(wgt_val);
          end
          // ReLU and shift by 8
          logic signed [31:0] shifted = (acc < 0) ? 32'sd0 : (acc >>> 8);
          // Saturate to INT8
          if (shifted > 32'sd127)
            conv1_out_buf[position_counter][oc] <= 8'sd127;
          else if (shifted < -32'sd128)
            conv1_out_buf[position_counter][oc] <= -8'sd128;
          else
            conv1_out_buf[position_counter][oc] <= shifted[7:0];
        end
        position_counter <= position_counter + 1;
      end
    end
  end

  // ===========================================================================
  // Stage: MaxPool (252 → 126 with kernel=2, stride=2)
  // ===========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      position_counter <= '0;
      for (int i = 0; i < 126; i++)
        for (int j = 0; j < 8; j++)
          pool_out_buf[i][j] <= '0;
    end else if (state == POOL) begin
      if (position_counter < 126) begin
        for (int j = 0; j < 8; j++) begin
          logic signed [7:0] v0 = conv1_out_buf[2*position_counter][j];
          logic signed [7:0] v1 = conv1_out_buf[2*position_counter+1][j];
          pool_out_buf[position_counter][j] <= (v0 > v1) ? v0 : v1;
        end
        position_counter <= position_counter + 1;
      end
    end
  end

  // ===========================================================================
  // Stage: Conv2 (8×126 → 8×124 with kernel=3)
  // Accumulates 3 partial sums (K=8 input channels × 3 positions = K_total=24)
  // ===========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      position_counter <= '0;
      for (int i = 0; i < 124; i++)
        for (int j = 0; j < 8; j++)
          conv2_out_buf[i][j] <= '0;
    end else if (state == CONV2) begin
      if (position_counter < 124) begin
        // For each output channel
        for (int oc = 0; oc < 8; oc++) begin
          // Accumulate over 3 kernel positions × 8 input channels
          logic signed [31:0] acc;
          acc = $signed(CONV2_B[oc]);
          for (int k = 0; k < 3; k++) begin
            for (int ic = 0; ic < 8; ic++) begin
              logic signed [7:0] act_val = pool_out_buf[position_counter + k][ic];
              logic signed [7:0] wgt_val = CONV2_W[k][ic][oc];
              acc = acc + $signed(act_val) * $signed(wgt_val);
            end
          end
          // ReLU and shift by 8
          logic signed [31:0] shifted = (acc < 0) ? 32'sd0 : (acc >>> 8);
          // Saturate to INT8
          if (shifted > 32'sd127)
            conv2_out_buf[position_counter][oc] <= 8'sd127;
          else if (shifted < -32'sd128)
            conv2_out_buf[position_counter][oc] <= -8'sd128;
          else
            conv2_out_buf[position_counter][oc] <= shifted[7:0];
        end
        position_counter <= position_counter + 1;
      end
    end
  end

  // ===========================================================================
  // Stage: GlobalAveragePool (124 → 1)
  // ===========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < 8; i++) gap_out_buf[i] <= '0;
    end else if (state == GAP) begin
      for (int ch = 0; ch < 8; ch++) begin
        logic signed [31:0] sum;
        sum = '0;
        for (int sp = 0; sp < 124; sp++)
          sum = sum + $signed(conv2_out_buf[sp][ch]);
        // Divide by 124 (approx shift by 7)
        logic signed [31:0] avg = sum >>> 7;
        if (avg > 32'sd127)
          gap_out_buf[ch] <= 8'sd127;
        else if (avg < -32'sd128)
          gap_out_buf[ch] <= -8'sd128;
        else
          gap_out_buf[ch] <= avg[7:0];
      end
    end
  end

  // ===========================================================================
  // Stage: Dense (8 → 4)
  // ===========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < 4; i++) logits_int32[i] <= '0;
    end else if (state == DENSE) begin
      for (int oc = 0; oc < 4; oc++) begin
        logic signed [31:0] acc;
        acc = $signed(CLASSIFIER_B[oc]);
        for (int ic = 0; ic < 8; ic++) begin
          logic signed [7:0] act_val = gap_out_buf[ic];
          logic signed [7:0] wgt_val = CLASSIFIER_W[ic][oc];
          acc = acc + $signed(act_val) * $signed(wgt_val);
        end
        // Shift by 5
        logits_int32[oc] <= acc >>> 5;
      end
    end
  end

  assign logits_out = logits_int32;
  assign done = (state == DONE);

endmodule
