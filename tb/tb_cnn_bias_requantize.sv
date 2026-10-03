` 1ns/1ps

module tb_cnn_bias_requantize;
  localparam LANES = 8;
  logic clk=0, rst_n=0, enable=0, relu_en=0;
  logic [4:0] shift_amount=0;
  logic signed [7:0] zero_point=0;
  logic [32*LANES-1:0] acc_in_packed='0;
  logic [32*LANES-1:0] bias_in_packed='0;
  logic acc_valid=0;
  logic [8*LANES-1:0] quant_out_packed;
  logic quant_valid, any_saturated;

  cnn_bias_requantize dut (
    .clk(clk), .rst_n(rst_n), .enable(enable), .relu_en(relu_en),
    .shift_amount(shift_amount), .zero_point(zero_point),
    .acc_in_packed(acc_in_packed), .bias_in_packed(bias_in_packed),
    .acc_valid(acc_valid), .quant_out_packed(quant_out_packed),
    .quant_valid(quant_valid), .any_saturated(any_saturated)
  );

  always #5 clk=~clk;

  task automatic set_lane(input integer lane, input integer acc, input integer bias);
    begin
      acc_in_packed[lane*32 +: 32]=acc;
      bias_in_packed[lane*32 +: 32]=bias;
    end
  endtask

  task automatic check_lane(input integer lane, input integer expected);
    integer got;
    begin
      got=$signed(quant_out_packed[lane*8 +: 8]);
      if (got !== expected) begin
        $display("FAIL lane=%0d got=%0d expected=%0d",lane,got,expected);
        $fatal;
      end
    end
  endtask

  initial begin
    repeat(2) @(posedge clk); rst_n=1;

    set_lane(0,1024,256); shift_amount=8; relu_en=0; enable=1; acc_valid=1;
    @(posedge clk); acc_valid=0; @(posedge clk); @(posedge clk);
    if (!quant_valid) begin $display("FAIL quant_valid"); $fatal; end
    check_lane(0,5);

    set_lane(1,-100,50); shift_amount=0; relu_en=1; acc_valid=1;
    @(posedge clk); acc_valid=0; @(posedge clk); @(posedge clk); check_lane(1,0);

    set_lane(2,200,100); shift_amount=0; relu_en=0; acc_valid=1;
    @(posedge clk); acc_valid=0; @(posedge clk); @(posedge clk); check_lane(2,127);

    set_lane(3,-200,-100); acc_valid=1;
    @(posedge clk); acc_valid=0; @(posedge clk); @(posedge clk); check_lane(3,-128);

    $display("CNN_BIAS_REQUANT_TEST PASSED"); $finish;
  end
endmodule
