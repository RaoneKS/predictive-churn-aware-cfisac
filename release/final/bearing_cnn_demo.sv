// =============================================================================
// Module: bearing_cnn_demo
// Application-specific fixed-model accelerator for the Phase-3 CWRU CNN.
//
// Architecture
//   - 64 parallel INT8 MAC datapaths (8 output lanes × 8 input lanes)
//   - Conv1: 8 output channels, 5 tap cycles, all 8 outputs in parallel
//   - MaxPool1D(2): all 8 channels in parallel
//   - Conv2: 8 output channels, 3 flattened-K blocks × 8 MACs/block
//   - GAP: signed truncating average over 128 samples
//   - Classifier: 8 input lanes × 8 output lanes in parallel
//
// This module is the board-level application engine.  The generic
// inference_accelerator_top remains in rtl/ as the reusable tiled accelerator
// baseline and is verified independently.
//
// Model arithmetic is taken directly from ml/hardware_reference.py:
// INT8 x INT8 -> INT32, bias before ReLU/shift, arithmetic right shift,
// INT8 saturation.  GAP uses truncation toward zero for signed sums.
//
// Board demo:
//   SW[1]=0 -> normal CWRU case
//   SW[1]=1 -> outer-race case
//   SW[0] rising edge -> start
// =============================================================================
module bearing_cnn_demo (
  input  logic       clk,
  input  logic       rst_n,
  input  logic       start,
  input  logic       select_outer,
  output logic       busy,
  output logic       done,
  output logic [1:0] predicted_class,
  output logic [3:0] state,
  output logic [31:0] logits_out
);

  localparam int IDLE = 0;
  localparam int C1   = 1;
  localparam int POOL = 2;
  localparam int C2   = 3;
  localparam int GAP  = 4;
  localparam int FC   = 5;
  localparam int PICK = 6;
  localparam int DONE = 7;

  logic [3:0] st;

  logic       select_latched;
  logic [8:0] pos;       // 0..255 for Conv1, 0..127 afterwards
  logic [2:0] tap;       // Conv1: 0..4

  // Conv1 timing pipeline:
  // phase 0 = address/multiply and register product
  // phase 1 = accumulate / quantize / store
  logic c1_phase;
  logic signed [31:0] c1_product_reg [0:7];

  // Time-multiplexed Conv2 counters.
  // One flattened-K element is processed per cycle.
  logic [2:0] c2_in_ch;   // 0..7
  logic [1:0] c2_tap;     // 0..2

  // Conv2 timing pipeline:
  // phase 0 = multiply and register product
  // phase 1 = accumulate / quantize / store
  logic c2_phase;
  logic signed [31:0] c2_product_reg [0:7];

  // Time-multiplexed classifier input counter.
  logic [2:0] fc_idx;     // 0..7

  // ---------------------------------------------------------------------------
  // Fixed model ROMs
  // ---------------------------------------------------------------------------
  logic signed [7:0]  input_rom [0:511];
  logic signed [7:0]  c1_w     [0:63];
  logic signed [7:0]  c2_w     [0:191];
  logic signed [7:0]  fc_w     [0:63];

  logic signed [31:0] c1_b [0:7];
  logic signed [31:0] c2_b [0:7];
  logic signed [31:0] fc_b [0:3];

  // ---------------------------------------------------------------------------
  // Intermediate feature memories
  // ---------------------------------------------------------------------------
  logic signed [7:0] c1_mem   [0:2047]; // 256 × 8
  logic signed [7:0] pool_mem [0:1023]; // 128 × 8
  logic signed [7:0] c2_mem   [0:1023]; // 128 × 8
  logic signed [7:0] gap_mem  [0:7];
  logic signed [7:0] logits   [0:7];

  // ---------------------------------------------------------------------------
  // 8 output accumulators.
  // Conv1 keeps 8 output lanes parallel.
  // Conv2 and FC reuse the same 8 multiplier lanes across cycles.
  // ---------------------------------------------------------------------------
  logic signed [31:0] c1_acc [0:7];
  logic signed [31:0] c2_acc [0:7];
  logic signed [31:0] fc_acc [0:7];
  logic signed [31:0] gap_acc[0:7];

  integer i;
  integer idx;
  integer sample_idx;
  integer kidx;
  integer tmp;
  logic signed [31:0] acc_tmp;
  logic signed [31:0] product_tmp;

  // ---------------------------------------------------------------------------
  // Exact deployment arithmetic
  // ---------------------------------------------------------------------------
  function automatic logic signed [31:0] mul8(
      input logic signed [7:0] a,
      input logic signed [7:0] b
  );
    logic signed [15:0] p;
    begin
      p = a * b;
      mul8 = {{16{p[15]}}, p};
    end
  endfunction

  function automatic logic signed [7:0] quantize32(
      input logic signed [31:0] value,
      input logic [4:0] shift,
      input logic relu_en
  );
    logic signed [31:0] tmp_q;
    begin
      tmp_q = relu_en && value[31] ? 32'sd0 : value;
      tmp_q = tmp_q >>> shift;

      if (tmp_q > 32'sd127)
        quantize32 = 8'sd127;
      else if (tmp_q < -32'sd128)
        quantize32 = -8'sd128;
      else
        quantize32 = tmp_q[7:0];
    end
  endfunction

  // Exact truncation toward zero for division by 128.
  function automatic logic signed [7:0] gap_div128(
      input logic signed [31:0] value
  );
    logic signed [31:0] mag;
    logic signed [31:0] q;
    begin
      if (value >= 0)
        q = value >>> 7;
      else begin
        mag = -value;
        q = -(mag >>> 7);
      end

      if (q > 32'sd127)
        gap_div128 = 8'sd127;
      else if (q < -32'sd128)
        gap_div128 = -8'sd128;
      else
        gap_div128 = q[7:0];
    end
  endfunction

  // ---------------------------------------------------------------------------
  // Model initialization
  // ---------------------------------------------------------------------------
  initial begin
    $readmemh("model_data/demo_inputs.hex",        input_rom);
    $readmemh("model_data/conv1_weights.hex",      c1_w);
    $readmemh("model_data/conv1_bias.hex",         c1_b);
    $readmemh("model_data/conv2_weights.hex",      c2_w);
    $readmemh("model_data/conv2_bias.hex",         c2_b);
    $readmemh("model_data/classifier_weights.hex", fc_w);
    $readmemh("model_data/classifier_bias.hex",    fc_b);
  end

  // ---------------------------------------------------------------------------
  // State machine
  // ---------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= IDLE;
      pos <= '0;
      tap <= '0;
      c1_phase <= 1'b0;
      c2_in_ch <= '0;
      c2_tap <= '0;
      c2_phase <= 1'b0;
      fc_idx <= '0;
      select_latched <= 1'b0;
      predicted_class <= '0;

      for (i = 0; i < 8; i = i + 1) begin
        c1_acc[i]  <= '0;
        c2_acc[i]  <= '0;
        fc_acc[i]  <= '0;
        gap_acc[i] <= '0;
        gap_mem[i] <= '0;
        logits[i]  <= '0;
      end
    end else begin
      case (st)

        // ---------------------------------------------------------------------
        // Conv1: 256 positions × 5 taps.
        // Same padding is implemented by returning zero outside [0,255].
        // Eight output channels are computed in parallel.
        // ---------------------------------------------------------------------
        C1: begin
          // Phase 0: calculate the input sample and INT8 product, then
          // register the product. This removes the multiplier and address
          // generation from the same cycle as accumulation/quantization.
          if (!c1_phase) begin
            for (i = 0; i < 8; i = i + 1) begin
              sample_idx = pos + tap;

              if ((sample_idx >= 2) && (sample_idx < 258))
                sample_idx = (select_latched ? 256 : 0) + sample_idx - 2;
              else
                sample_idx = -1;

              if (sample_idx >= 0)
                product_tmp = mul8(input_rom[sample_idx], c1_w[tap*8+i]);
              else
                product_tmp = 32'sd0;

              c1_product_reg[i] <= product_tmp;
            end

            c1_phase <= 1'b1;
          end

          // Phase 1: accumulate the registered products and finish the
          // current tap.
          else begin
            for (i = 0; i < 8; i = i + 1) begin
              acc_tmp = c1_acc[i] + c1_product_reg[i];

              if (tap == 4) begin
                c1_mem[pos*8+i] <=
                  quantize32(acc_tmp + c1_b[i], 5'd8, 1'b1);
                c1_acc[i] <= '0;
              end
              else begin
                c1_acc[i] <= acc_tmp;
              end
            end

            c1_phase <= 1'b0;

            if (tap == 4) begin
              tap <= '0;

              if (pos == 255) begin
                pos <= '0;
                st <= POOL;
              end
              else begin
                pos <= pos + 1'b1;
              end
            end
            else begin
              tap <= tap + 1'b1;
            end
          end
        end

        POOL: begin
          for (i = 0; i < 8; i = i + 1) begin
            if (c1_mem[(pos*2)*8+i] >= c1_mem[(pos*2+1)*8+i])
              pool_mem[pos*8+i] <= c1_mem[(pos*2)*8+i];
            else
              pool_mem[pos*8+i] <= c1_mem[(pos*2+1)*8+i];
          end

          if (pos == 127) begin
            pos <= '0;
            c2_in_ch <= '0;
            c2_tap <= '0;
            c2_phase <= 1'b0;
            for (i = 0; i < 8; i = i + 1)
              c2_acc[i] <= '0;
            st <= C2;
          end else begin
            pos <= pos + 1'b1;
          end
        end

        // ---------------------------------------------------------------------
        // Conv2: K=24 = three blocks of eight input-channel/tap elements.
        // 64 products are formed per cycle: 8 input lanes × 8 output lanes.
        // Flattened K ordering is channel-major then kernel tap, matching
        // hardware_reference.py.
        // ---------------------------------------------------------------------
        // ---------------------------------------------------------------------
        // Conv2: K=24, time-multiplexed.
        //
        // One flattened-K element is processed per cycle while all 8
        // output channels remain parallel.
        //
        // This changes the hardware from:
        //   64 multipliers/cycle
        // to:
        //   8 reused multipliers/cycle.
        //
        // Flattened K ordering:
        //   input_channel 0..7
        //   kernel tap     0..2
        //
        // Padding exactly matches the previous implementation.
        // ---------------------------------------------------------------------
        C2: begin
          // Phase 0: perform the INT8 multiply and register the products.
          // This breaks the critical path before the accumulator/quantizer.
          if (!c2_phase) begin
            for (i = 0; i < 8; i = i + 1) begin
              sample_idx = pos + c2_tap;

              if ((sample_idx >= 1) && (sample_idx < 129))
                product_tmp =
                  mul8(
                    pool_mem[(sample_idx - 1) * 8 + c2_in_ch],
                    c2_w[(c2_in_ch * 3 + c2_tap) * 8 + i]
                  );
              else
                product_tmp = 32'sd0;

              c2_product_reg[i] <= product_tmp;
            end

            c2_phase <= 1'b1;
          end

          // Phase 1: consume the registered products.
          else begin
            for (i = 0; i < 8; i = i + 1) begin
              acc_tmp = c2_acc[i] + c2_product_reg[i];

              if ((c2_in_ch == 3'd7) && (c2_tap == 2'd2)) begin
                c2_mem[pos * 8 + i] <=
                  quantize32(acc_tmp + c2_b[i], 5'd8, 1'b1);
                c2_acc[i] <= '0;
              end
              else begin
                c2_acc[i] <= acc_tmp;
              end
            end

            c2_phase <= 1'b0;

            if ((c2_in_ch == 3'd7) && (c2_tap == 2'd2)) begin
              c2_in_ch <= '0;
              c2_tap <= '0;

              if (pos == 127) begin
                pos <= '0;

                for (i = 0; i < 8; i = i + 1)
                  gap_acc[i] <= '0;

                st <= GAP;
              end
              else begin
                pos <= pos + 1'b1;
              end
            end
            else if (c2_tap == 2'd2) begin
              c2_tap <= '0;
              c2_in_ch <= c2_in_ch + 1'b1;
            end
            else begin
              c2_tap <= c2_tap + 1'b1;
            end
          end
        end

        GAP: begin
          for (i = 0; i < 8; i = i + 1) begin
            acc_tmp = gap_acc[i] + $signed(c2_mem[pos*8+i]);
            if (pos == 127) begin
              gap_mem[i] <= gap_div128(acc_tmp);
              gap_acc[i] <= '0;
            end else begin
              gap_acc[i] <= acc_tmp;
            end
          end

          if (pos == 127) begin
            pos <= '0;
            fc_idx <= '0;
            for (i = 0; i < 8; i = i + 1)
              fc_acc[i] <= '0;
            st <= FC;
          end else begin
            pos <= pos + 1'b1;
          end
        end

        // ---------------------------------------------------------------------
        // Classifier: all 8 input lanes × 8 output lanes in one cycle.
        // Only logits 0..3 are used for the final class.
        // ---------------------------------------------------------------------
        // ---------------------------------------------------------------------
        // Classifier: K=8, time-multiplexed.
        //
        // One input feature is processed per cycle while all 8 output
        // lanes remain parallel.
        //
        // This changes the hardware from:
        //   64 multipliers in one cycle
        // to:
        //   8 reused multipliers.
        // ---------------------------------------------------------------------
        FC: begin
          for (i = 0; i < 8; i = i + 1) begin
            product_tmp =
              mul8(gap_mem[fc_idx], fc_w[fc_idx * 8 + i]);

            acc_tmp = fc_acc[i] + product_tmp;

            if (fc_idx == 3'd7) begin
              if (i < 4)
                logits[i] <=
                  quantize32(acc_tmp + fc_b[i], 5'd5, 1'b0);
              else
                logits[i] <=
                  quantize32(acc_tmp, 5'd5, 1'b0);

              fc_acc[i] <= '0;
            end
            else begin
              fc_acc[i] <= acc_tmp;
            end
          end

          if (fc_idx == 3'd7) begin
            fc_idx <= '0;
            st <= PICK;
          end
          else begin
            fc_idx <= fc_idx + 1'b1;
          end
        end

        PICK: begin
          if ((logits[1] > logits[0]) &&
              (logits[1] >= logits[2]) &&
              (logits[1] >= logits[3]))
            predicted_class <= 2'd1;
          else if ((logits[2] > logits[0]) &&
                   (logits[2] > logits[1]) &&
                   (logits[2] >= logits[3]))
            predicted_class <= 2'd2;
          else if ((logits[3] > logits[0]) &&
                   (logits[3] > logits[1]) &&
                   (logits[3] > logits[2]))
            predicted_class <= 2'd3;
          else
            predicted_class <= 2'd0;
          st <= DONE;
        end

        DONE: begin
          if (!start)
            st <= IDLE;
        end

        IDLE: begin
          if (start) begin
            select_latched <= select_outer;
            pos <= '0;
            tap <= '0;
            c1_phase <= 1'b0;
            c2_in_ch <= '0;
            c2_tap <= '0;
            fc_idx <= '0;
            for (i = 0; i < 8; i = i + 1) begin
              c1_acc[i]  <= '0;
              c2_acc[i]  <= '0;
              fc_acc[i]  <= '0;
              gap_acc[i] <= '0;
            end
            st <= C1;
          end
        end

        default: st <= IDLE;
      endcase
    end
  end

  assign busy = (st != IDLE) && (st != DONE);
  assign done = (st == DONE);
  assign state = st;
  assign logits_out = {logits[3], logits[2], logits[1], logits[0]};

endmodule
