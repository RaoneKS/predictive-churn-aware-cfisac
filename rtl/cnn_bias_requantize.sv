// Bias-aware INT32 -> INT8 requantization for the CNN integration path.
// Additive module: validated generic quantize_unit/top interfaces remain unchanged.
//
// Arithmetic:
//   acc + bias -> optional ReLU -> arithmetic right shift -> zero point -> INT8 saturation

`timescale 1ns/1ps

import pkg_accelerator::*;

module cnn_bias_requantize #(
  parameter LANES = ARRAY_COLS
)(
  input  logic clk,
  input  logic rst_n,
  input  logic enable,
  input  logic relu_en,
  input  logic [4:0] shift_amount,
  input  logic signed [OUT_WIDTH-1:0] zero_point,
  input  logic [ACC_WIDTH*LANES-1:0] acc_in_packed,
  input  logic [ACC_WIDTH*LANES-1:0] bias_in_packed,
  input  logic acc_valid,
  output logic [OUT_WIDTH*LANES-1:0] quant_out_packed,
  output logic quant_valid,
  output logic any_saturated
);

  logic signed [ACC_WIDTH-1:0] stage1_data [LANES];
  logic stage1_valid;

  genvar i;
  generate
    for (i = 0; i < LANES; i = i + 1) begin : gen_stage1
      logic signed [ACC_WIDTH-1:0] acc_i;
      logic signed [ACC_WIDTH-1:0] bias_i;
      logic signed [ACC_WIDTH-1:0] biased_i;

      assign acc_i = $signed(acc_in_packed[i*ACC_WIDTH +: ACC_WIDTH]);
      assign bias_i = $signed(bias_in_packed[i*ACC_WIDTH +: ACC_WIDTH]);
      assign biased_i = acc_i + bias_i;

      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
          stage1_data[i] <= '0;
        else if (enable && acc_valid)
          stage1_data[i] <= (relu_en && biased_i[ACC_WIDTH-1]) ? '0 : biased_i;
      end
    end
  endgenerate

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      stage1_valid <= 1'b0;
    else if (enable)
      stage1_valid <= acc_valid;
    else
      stage1_valid <= 1'b0;
  end

  logic signed [OUT_WIDTH-1:0] stage2_data [LANES];
  logic stage2_valid;
  logic [LANES-1:0] sat_flags;

  generate
    for (i = 0; i < LANES; i = i + 1) begin : gen_stage2
      logic signed [ACC_WIDTH-1:0] shifted;
      logic signed [ACC_WIDTH-1:0] with_zp;
      logic saturated;

      assign shifted = stage1_data[i] >>> shift_amount;
      assign with_zp = shifted +
        {{(ACC_WIDTH-OUT_WIDTH){zero_point[OUT_WIDTH-1]}}, zero_point};
      assign saturated =
        (with_zp > 32'sd127) || (with_zp < -32'sd128);

      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          stage2_data[i] <= '0;
          sat_flags[i] <= 1'b0;
        end else if (enable && stage1_valid) begin
          if (with_zp > 32'sd127)
            stage2_data[i] <= 8'sd127;
          else if (with_zp < -32'sd128)
            stage2_data[i] <= -8'sd128;
          else
            stage2_data[i] <= with_zp[OUT_WIDTH-1:0];
          sat_flags[i] <= saturated;
        end else begin
          sat_flags[i] <= 1'b0;
        end
      end
    end
  endgenerate

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      stage2_valid <= 1'b0;
    else if (enable)
      stage2_valid <= stage1_valid;
    else
      stage2_valid <= 1'b0;
  end

  generate
    for (i = 0; i < LANES; i = i + 1) begin : gen_output
      assign quant_out_packed[i*OUT_WIDTH +: OUT_WIDTH] = stage2_data[i];
    end
  endgenerate

  assign quant_valid = stage2_valid;
  assign any_saturated = |sat_flags;

endmodule
