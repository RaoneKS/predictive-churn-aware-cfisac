// Board-facing Phase-3 predictive-maintenance demonstration.
// KEY[0] reset; pulse SW[0] high to start; SW[1]=0 normal, SW[1]=1 outer fault.
module de10_standard_hw_top (
  input logic CLOCK_50,
  input logic [0:0] KEY,
  input logic [1:0] SW,
  output logic [9:0] LEDR
);
  logic [25:0] heartbeat;
  logic sw0_d, start_pulse;
  logic busy, done;
  logic [1:0] predicted_class;
  logic [3:0] state;
  logic [31:0] logits_unused;

  always_ff @(posedge CLOCK_50 or negedge KEY[0]) begin
    if (!KEY[0]) begin
      heartbeat <= '0;
      sw0_d <= 1'b0;
      start_pulse <= 1'b0;
    end else begin
      heartbeat <= heartbeat + 1'b1;
      start_pulse <= SW[0] && !sw0_d && !busy;
      sw0_d <= SW[0];
    end
  end

  bearing_cnn_demo u_demo (
    .clk(CLOCK_50), .rst_n(KEY[0]), .start(start_pulse), .select_outer(SW[1]),
    .busy(busy), .done(done), .predicted_class(predicted_class), .state(state),
    .logits_out(logits_unused)
  );

  assign LEDR[0] = heartbeat[25];
  assign LEDR[1] = busy;
  assign LEDR[2] = done;
  assign LEDR[5:3] = {1'b0, predicted_class};
  assign LEDR[9:6] = state;
endmodule
