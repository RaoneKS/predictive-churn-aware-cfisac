`timescale 1ns/1ps
module tb_bearing_cnn_demo;
  logic clk=0, rst_n=0, start=0, select_outer=0;
  logic busy, done; logic [1:0] predicted_class; logic [3:0] state; logic [31:0] logits_out;
  bearing_cnn_demo dut(.*);
  always #10 clk=~clk;
  task automatic run_case(input logic fault, input logic [1:0] expected,
                          input integer l0,input integer l1,input integer l2,input integer l3);
    integer timeout;
    begin
      select_outer=fault;
      @(posedge clk); start=1;
      @(posedge clk); start=0;
      timeout=0;
      while (!done && timeout<50000) begin @(posedge clk); timeout=timeout+1; end
      if (!done) begin $display("FAIL timeout state=%0d",state); $fatal; end
      if (predicted_class!==expected) begin
        $display("FAIL select_outer=%0d got=%0d expected=%0d",fault,predicted_class,expected); $fatal;
      end
      if ($signed(logits_out[7:0])!==l0 || $signed(logits_out[15:8])!==l1 ||
          $signed(logits_out[23:16])!==l2 || $signed(logits_out[31:24])!==l3) begin
        $display("FAIL logits got %0d %0d %0d %0d", $signed(logits_out[7:0]),
          $signed(logits_out[15:8]),$signed(logits_out[23:16]),$signed(logits_out[31:24])); $fatal;
      end
      $display("PASS select_outer=%0d class=%0d cycles=%0d",fault,predicted_class,timeout);
      @(posedge clk);
    end
  endtask
  initial begin
    repeat(4) @(posedge clk); rst_n=1;
    run_case(1'b0,2'd0,39,2,-36,-51);
    run_case(1'b1,2'd3,-33,3,9,15);
    $display("BEARING_CNN_DEMO_TEST PASSED"); $finish;
  end
endmodule
