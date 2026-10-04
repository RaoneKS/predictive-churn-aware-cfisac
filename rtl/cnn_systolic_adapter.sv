// ===========================================================================
// Module:  cnn_systolic_adapter.sv
// Purpose: CNN-specific wrapper around systolic_array.sv
//          Handles Conv1/Conv2/Classifier im2col scheduling and tile accumulation
// ===========================================================================

`timescale 1ns / 1ps

import pkg_accelerator::*;

module cnn_systolic_adapter (
  input  logic                          clk,
  input  logic                          rst_n,
  input  logic                          enable,

  // Activation input (8 lanes, one per row)
  input  logic signed [7:0]             act_in [ARRAY_ROWS],
  // Weight input (8 lanes, one per column)
  input  logic signed [7:0]             wgt_in [ARRAY_COLS],
  // Valid signal for activation
  input  logic                          act_valid,

  // Accumulator output (8 lanes, one per column)
  output logic signed [31:0]            acc_out [ARRAY_COLS],
  output logic                          acc_valid,
  output logic                          any_overflow
);

  // Pack/unpack signals for systolic array
  logic [ACT_WIDTH*ARRAY_ROWS-1:0] act_packed;
  logic [ACT_WIDTH*ARRAY_COLS-1:0] acc_packed;

  // Pack activation inputs
  genvar i;
  generate
    for (i = 0; i < ARRAY_ROWS; i = i + 1) begin : gen_act_pack
      assign act_packed[i*ACT_WIDTH +: ACT_WIDTH] = act_in[i];
    end
  endgenerate

  // Pack weights into array format
  logic signed [WGT_WIDTH-1:0] wgt_array [ARRAY_COLS];
  generate
    for (i = 0; i < ARRAY_COLS; i = i + 1) begin : gen_wgt_pack
      assign wgt_array[i] = wgt_in[i];
    end
  endgenerate

  // Instantiate systolic array
  systolic_array u_systolic (
    .clk               (clk),
    .rst_n             (rst_n),
    .enable            (enable),
    .clear_acc         (1'b0),          // Never clear during normal operation
    .valid_reset       (1'b0),          // Never reset validity counter
    .act_in_packed     (act_packed),
    .wgt_in            (wgt_array),
    .data_valid_in     (act_valid),
    .data_valid_out    (acc_valid),
    .acc_out_packed    (acc_packed),
    .any_overflow      (any_overflow)
  );

  // Unpack accumulator outputs
  generate
    for (i = 0; i < ARRAY_COLS; i = i + 1) begin : gen_acc_unpack
      assign acc_out[i] = $signed(acc_packed[i*ACC_WIDTH +: ACC_WIDTH]);
    end
  endgenerate

endmodule
