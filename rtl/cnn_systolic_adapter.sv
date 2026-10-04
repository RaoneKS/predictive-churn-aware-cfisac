// Correctness-first 8x8 systolic tile wrapper.
// Feeds one K element per cycle. Row k receives activation[k] and weight row k.
// The systolic array itself supplies the horizontal/column skew.

`timescale 1ns/1ps
import pkg_accelerator::*;

module cnn_systolic_adapter (
  input  logic clk,
  input  logic rst_n,
  input  logic start,
  input  logic [ACT_WIDTH*ARRAY_ROWS-1:0] act_vector,
  input  logic [WGT_WIDTH*ARRAY_ROWS-1:0] weight_rows [ARRAY_ROWS],
  output logic busy,
  output logic done,
  output logic [ACC_WIDTH*ARRAY_COLS-1:0] acc_out_packed,
  output logic any_overflow
);

  typedef enum logic [2:0] {S_IDLE,S_CLEAR,S_FEED,S_DRAIN,S_DONE} state_t;
  state_t state;
  logic [3:0] k_index;
  logic [5:0] drain_count;

  logic [ACT_WIDTH*ARRAY_ROWS-1:0] act_packed;
  logic [WGT_WIDTH-1:0] wgt_cols [ARRAY_COLS];
  logic [ACC_WIDTH*ARRAY_COLS-1:0] array_acc;
  logic array_valid;
  logic array_overflow;

  genvar c;
  generate
    for (c=0;c<ARRAY_COLS;c=c+1) begin : gen_w
      assign wgt_cols[c] = weight_rows[k_index][c*WGT_WIDTH +: WGT_WIDTH];
    end
  endgenerate

  always_comb begin
    act_packed = '0;
    if (state == S_FEED)
      act_packed[k_index*ACT_WIDTH +: ACT_WIDTH] =
        act_vector[k_index*ACT_WIDTH +: ACT_WIDTH];
  end

  systolic_array u_systolic (
    .clk(clk), .rst_n(rst_n), .enable(1'b1),
    .clear_acc(state == S_CLEAR),
    .valid_reset(state == S_CLEAR),
    .act_in_packed(act_packed),
    .wgt_in(wgt_cols),
    .data_valid_in(state == S_FEED),
    .data_valid_out(array_valid),
    .acc_out_packed(array_acc),
    .any_overflow(array_overflow)
  );

  assign busy = (state != S_IDLE);
  assign done = (state == S_DONE);
  assign any_overflow = array_overflow;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE;
      k_index <= '0;
      drain_count <= '0;
      acc_out_packed <= '0;
    end else begin
      case (state)
        S_IDLE: if (start) begin k_index <= 0; state <= S_CLEAR; end
        S_CLEAR: begin k_index <= 0; state <= S_FEED; end
        S_FEED: begin
          if (k_index == ARRAY_ROWS-1) begin
            drain_count <= 0;
            state <= S_DRAIN;
          end else k_index <= k_index + 1'b1;
        end
        S_DRAIN: begin
          if (array_valid) begin
            acc_out_packed <= array_acc;
            state <= S_DONE;
          end else if (drain_count == 63) begin
            acc_out_packed <= array_acc;
            state <= S_DONE;
          end else drain_count <= drain_count + 1'b1;
        end
        S_DONE: state <= S_IDLE;
        default: state <= S_IDLE;
      endcase
    end
  end
endmodule
