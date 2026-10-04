// ===========================================================================
// Module:  cnn_inference_top.sv
// Purpose: CNN end-to-end inference (Conv1 -> Pool -> Conv2 -> GAP -> Classifier)
//          Reuses systolic array via cnn_systolic_adapter
//          Loads authoritative weights from testbench memory interface
// ===========================================================================

`timescale 1ns / 1ps

import pkg_accelerator::*;

module cnn_inference_top (
  input  logic                          clk,
  input  logic                          rst_n,
  input  logic                          start,
  output logic                          done,

  // Window input (256 INT8 samples, one per cycle)
  input  logic signed [7:0]             window_sample,
  input  logic                          window_valid,
  output logic                          window_ready,

  // Weights/biases from testbench
  input  logic signed [7:0]             conv1_weights [5][8],
  input  logic signed [31:0]            conv1_bias [8],
  input  logic signed [7:0]             conv2_weights [3][8][8],
  input  logic signed [31:0]            conv2_bias [8],
  input  logic signed [7:0]             classifier_weights [8][4],
  input  logic signed [31:0]            classifier_bias [4],

  // Output logits (INT32 accumulator scale)
  output logic signed [31:0]            logits_out [4],
  output logic                          logits_valid,

  // Status
  output logic [31:0]                   cycle_count
);

  // ===========================================================================
  // Parameters from manifest
  // ===========================================================================
  localparam CONV1_SHIFT = 8;
  localparam CONV1_LEN = 256;
  localparam POOL_LEN = 128;
  localparam CONV2_SHIFT = 8;
  localparam CONV2_LEN = 128;
  localparam CLASSIFIER_SHIFT = 5;

  // ===========================================================================
  // Buffering
  // ===========================================================================
  logic signed [7:0] window_buf [256];
  logic [15:0] window_idx;
  logic window_buf_full;

  logic signed [7:0] conv1_out [256][8];
  logic signed [7:0] pool_out [128][8];
  logic signed [7:0] conv2_out [128][8];
  logic signed [7:0] gap_out [8];
  logic signed [31:0] logits [4];

  // ===========================================================================
  // State machine
  // ===========================================================================
  enum logic [3:0] {
    IDLE = 0,
    INPUT = 1,
    CONV1 = 2,
    POOL = 3,
    CONV2 = 4,
    GAP = 5,
    CLASSIFIER = 6,
    DONE = 7
  } state, next_state;

  logic [15:0] position;
  logic [31:0] cycle_ctr;

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
  // FSM
  // ===========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      position <= '0;
      window_buf_full <= 1'b0;
    end else begin
      state <= next_state;
    end
  end

  always_comb begin
    next_state = state;
    window_ready = 1'b0;

    case (state)
      IDLE: begin
        window_ready = 1'b1;
        if (start) next_state = INPUT;
      end

      INPUT: begin
        window_ready = 1'b1;
        if (window_buf_full) next_state = CONV1;
      end

      CONV1: if (position >= 255) next_state = POOL;
      POOL: if (position >= 127) next_state = CONV2;
      CONV2: if (position >= 127) next_state = GAP;
      GAP: next_state = CLASSIFIER;
      CLASSIFIER: next_state = DONE;
      DONE: next_state = IDLE;
    endcase
  end

  // ===========================================================================
  // Stage: Input buffering
  // ===========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      window_idx <= '0;
      window_buf_full <= 1'b0;
    end else begin
      if (state == IDLE) begin
        window_idx <= '0;
        window_buf_full <= 1'b0;
      end else if (state == INPUT && window_valid) begin
        window_buf[window_idx] <= window_sample;
        if (window_idx == 255) begin
          window_buf_full <= 1'b1;
        end else begin
          window_idx <= window_idx + 1;
        end
      end
    end
  end

  // ===========================================================================
  // Stage: Conv1 (256 positions, K=5)
  // ===========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      position <= '0;
      for (int i = 0; i < 256; i++)
        for (int j = 0; j < 8; j++)
          conv1_out[i][j] <= '0;
    end else if (state == CONV1) begin
      if (position < 256) begin
        // Compute MAC for this position
        for (int oc = 0; oc < 8; oc++) begin
          logic signed [31:0] acc;
          acc = $signed(conv1_bias[oc]);
          for (int k = 0; k < 5; k++) begin
            if ((position + k) < 256) begin
              logic signed [7:0] act_val = window_buf[position + k];
              logic signed [7:0] wgt_val = conv1_weights[k][oc];
              acc = acc + $signed(act_val) * $signed(wgt_val);
            end
          end
          // ReLU
          if (acc < 0) acc = 32'sd0;
          // Shift by 8
          logic signed [31:0] shifted = acc >>> CONV1_SHIFT;
          // Saturate
          if (shifted > 32'sd127)
            conv1_out[position][oc] <= 8'sd127;
          else if (shifted < -32'sd128)
            conv1_out[position][oc] <= -8'sd128;
          else
            conv1_out[position][oc] <= shifted[7:0];
        end
        position <= position + 1;
      end
    end
  end

  // ===========================================================================
  // Stage: MaxPool (256 -> 128, kernel=2, stride=2)
  // ===========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      position <= '0;
      for (int i = 0; i < 128; i++)
        for (int j = 0; j < 8; j++)
          pool_out[i][j] <= '0;
    end else if (state == POOL) begin
      if (position < 128) begin
        for (int j = 0; j < 8; j++) begin
          logic signed [7:0] v0 = conv1_out[2*position][j];
          logic signed [7:0] v1 = conv1_out[2*position + 1][j];
          pool_out[position][j] <= (v0 > v1) ? v0 : v1;
        end
        position <= position + 1;
      end
    end
  end

  // ===========================================================================
  // Stage: Conv2 (128 positions, K=24 = 3 tiles × 8)
  // Accumulates three K=8 partial results BEFORE bias/ReLU/shift
  // ===========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      position <= '0;
      for (int i = 0; i < 128; i++)
        for (int j = 0; j < 8; j++)
          conv2_out[i][j] <= '0;
    end else if (state == CONV2) begin
      if (position < 128) begin
        for (int oc = 0; oc < 8; oc++) begin
          logic signed [31:0] acc;
          acc = $signed(conv2_bias[oc]);
          // Three K=8 tiles: k=0..2 spatial positions, ic=0..7 input channels
          for (int k = 0; k < 3; k++) begin
            for (int ic = 0; ic < 8; ic++) begin
              logic signed [7:0] act_val = pool_out[position + k][ic];
              logic signed [7:0] wgt_val = conv2_weights[k][ic][oc];
              acc = acc + $signed(act_val) * $signed(wgt_val);
            end
          end
          // ReLU
          if (acc < 0) acc = 32'sd0;
          // Shift by 8
          logic signed [31:0] shifted = acc >>> CONV2_SHIFT;
          // Saturate
          if (shifted > 32'sd127)
            conv2_out[position][oc] <= 8'sd127;
          else if (shifted < -32'sd128)
            conv2_out[position][oc] <= -8'sd128;
          else
            conv2_out[position][oc] <= shifted[7:0];
        end
        position <= position + 1;
      end
    end
  end

  // ===========================================================================
  // Stage: GlobalAveragePool (128 samples -> 1, signed truncation toward zero)
  // ===========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < 8; i++) gap_out[i] <= '0;
    end else if (state == GAP) begin
      for (int ch = 0; ch < 8; ch++) begin
        logic signed [31:0] sum;
        sum = '0;
        for (int sp = 0; sp < 128; sp++)
          sum = sum + $signed(conv2_out[sp][ch]);
        // Divide by 128 = shift by 7, signed truncation toward zero
        logic signed [31:0] avg;
        if (sum >= 0)
          avg = sum >>> 7;
        else
          avg = -((-sum) >>> 7);
        // Saturate
        if (avg > 32'sd127)
          gap_out[ch] <= 8'sd127;
        else if (avg < -32'sd128)
          gap_out[ch] <= -8'sd128;
        else
          gap_out[ch] <= avg[7:0];
      end
    end
  end

  // ===========================================================================
  // Stage: Classifier (K=8 -> 4 logits)
  // ===========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < 4; i++) logits[i] <= '0;
    end else if (state == CLASSIFIER) begin
      for (int oc = 0; oc < 4; oc++) begin
        logic signed [31:0] acc;
        acc = $signed(classifier_bias[oc]);
        for (int ic = 0; ic < 8; ic++) begin
          logic signed [7:0] act_val = gap_out[ic];
          logic signed [7:0] wgt_val = classifier_weights[ic][oc];
          acc = acc + $signed(act_val) * $signed(wgt_val);
        end
        // Shift by 5 (no ReLU)
        logits[oc] <= acc >>> CLASSIFIER_SHIFT;
      end
    end
  end

  assign logits_out = logits;
  assign logits_valid = (state == DONE);
  assign done = (state == DONE);

endmodule
