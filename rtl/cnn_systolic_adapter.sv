// CNN single-vector tile adapter for the validated 8x8 systolic array.
// One tile computes one output position:
//   - activation vector: 8 INT8 K-elements
//   - weight rows: 8 cycles x 8 output channels
// The array's diagonal activation/weight skew aligns K rows.
// This wrapper intentionally favors correctness over throughput.

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

  localparam integer FINAL_CYCLE = ARRAY_LATENCY + (ARRAY_ROWS-1); // 25

  typedef enum logic [2:0] {S_IDLE, S_CLEAR, S_FEED, S_WAIT, S_DONE} state_t;
  state_t state;

  logic [ACT_WIDTH*ARRAY_ROWS-1:0] act_reg;
  logic [WGT_WIDTH*ARRAY_ROWS-1:0] weight_reg [ARRAY_ROWS];
  logic [3:0] weight_index;
  logic [5:0] wait_count;

  logic [WGT_WIDTH-1:0] wgt_cols [ARRAY_COLS];
  logic [ACT_WIDTH*ARRAY_ROWS-1:0] array_act;
  logic array_valid_in;
  logic array_clear;
  logic array_valid_reset;
  logic [ACC_WIDTH*ARRAY_COLS-1:0] array_acc;
  logic array_valid_out;
  logic array_overflow;

  genvar g;
  generate
    for (g = 0; g < ARRAY_COLS; g = g + 1) begin : gen_wcols
      assign wgt_cols[g] =
        weight_reg[weight_index][g*WGT_WIDTH +: WGT_WIDTH];
    end
  endgenerate

  assign array_act = act_reg;

  systolic_array u_systolic (
    .clk(clk),
    .rst_n(rst_n),
    .enable(1'b1),
    .clear_acc(array_clear),
    .valid_reset(array_valid_reset),
    .act_in_packed(array_act),
    .wgt_in(wgt_cols),
    .data_valid_in(array_valid_in),
    .data_valid_out(array_valid_out),
    .acc_out_packed(array_acc),
    .any_overflow(array_overflow)
  );

  assign busy = (state != S_IDLE);
  assign done = (state == S_DONE);
  assign any_overflow = array_overflow;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE;
      act_reg <= '0;
      weight_index <= '0;
      wait_count <= '0;
      acc_out_packed <= '0;
    end else begin
      case (state)
        S_IDLE: begin
          if (start) begin
            act_reg <= act_vector;
            for (integer r = 0; r < ARRAY_ROWS; r = r + 1)
              weight_reg[r] <= weight_rows[r];
            state <= S_CLEAR;
          end
        end

        S_CLEAR: begin
          // clear_acc/valid_reset are asserted combinationally for this state.
          weight_index <= 0;
          wait_count <= 0;
          state <= S_FEED;
        end

        S_FEED: begin
          // First K row enters the array with the activation vector.
          weight_index <= 0;
          wait_count <= 0;
          state <= S_WAIT;
        end

        S_WAIT: begin
          if (weight_index < ARRAY_ROWS-1) begin
            weight_index <= weight_index + 1'b1;
          end
          if (wait_count == FINAL_CYCLE) begin
            acc_out_packed <= array_acc;
            state <= S_DONE;
          end else begin
            wait_count <= wait_count + 1'b1;
          end
        end

        S_DONE: begin
          state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

  always_comb begin
    array_clear = (state == S_CLEAR);
    array_valid_reset = (state == S_CLEAR);
    array_valid_in = (state == S_FEED);
  end

endmodule
