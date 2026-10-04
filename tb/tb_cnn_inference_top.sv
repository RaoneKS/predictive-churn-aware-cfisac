`timescale 1ns/1ps
import pkg_accelerator::*;

module tb_cnn_inference_top;
  localparam integer N=512;
  localparam integer W=256;

  logic clk,rst_n,start,done,window_valid,window_ready,logits_valid;
  logic signed [7:0] window_sample;
  logic signed [31:0] logits_out[4];
  logic [31:0] cycle_count;

  logic signed [7:0] c1w_mem[0:39];
  logic signed [31:0] c1b_mem[0:7];
  logic signed [7:0] c2w_mem[0:191];
  logic signed [31:0] c2b_mem[0:7];
  logic signed [7:0] clfw_mem[0:31];
  logic signed [31:0] clfb_mem[0:3];

  logic signed [7:0] in_mem[0:N*W-1];
  logic signed [31:0] ref_mem[0:N*4-1];
  logic [7:0] pred_mem[0:N-1];
  logic [7:0] label_mem[0:N-1];

  logic signed [7:0] c1w[5][8];
  logic signed [31:0] c1b[8];
  logic signed [7:0] c2w[3][8][8];
  logic signed [31:0] c2b[8];
  logic signed [7:0] clfw[8][4];
  logic signed [31:0] clfb[4];

  integer i,j,k,w,s,c, matches,pred_matches,err_count;
  integer got_pred, best;
  integer got, refv;
  integer class_total [0:3];
  integer class_correct [0:3];
  integer ref_pred, ref_label;

  cnn_inference_top dut(
    .clk(clk),.rst_n(rst_n),.start(start),.done(done),
    .window_sample(window_sample),.window_valid(window_valid),.window_ready(window_ready),
    .conv1_weights(c1w),.conv1_bias(c1b),
    .conv2_weights(c2w),.conv2_bias(c2b),
    .classifier_weights(clfw),.classifier_bias(clfb),
    .logits_out(logits_out),.logits_valid(logits_valid),.cycle_count(cycle_count)
  );

  initial begin clk=0; forever #5 clk=~clk; end

  initial begin
    $readmemh("sim/cnn_vectors/input.hex",in_mem);
    $readmemh("sim/cnn_vectors/reference_logits.hex",ref_mem);
    $readmemh("sim/cnn_vectors/reference_predictions.hex",pred_mem);
    $readmemh("sim/cnn_vectors/reference_labels.hex",label_mem);
    $readmemh("sim/cnn_vectors/conv1_weights.hex",c1w_mem);
    $readmemh("sim/cnn_vectors/conv1_bias.hex",c1b_mem);
    $readmemh("sim/cnn_vectors/conv2_weights.hex",c2w_mem);
    $readmemh("sim/cnn_vectors/conv2_bias.hex",c2b_mem);
    $readmemh("sim/cnn_vectors/classifier_weights.hex",clfw_mem);
    $readmemh("sim/cnn_vectors/classifier_bias.hex",clfb_mem);

    for(k=0;k<8;k=k+1) begin
      c1b[k]=c1b_mem[k]; c2b[k]=c2b_mem[k];
    end
    for(c=0;c<4;c=c+1) clfb[c]=clfb_mem[c];

    for(k=0;k<5;k=k+1)
      for(c=0;c<8;c=c+1) c1w[k][c]=c1w_mem[c*5+k];

    for(k=0;k<3;k=k+1)
      for(j=0;j<8;j=j+1)
        for(c=0;c<8;c=c+1) c2w[k][j][c]=c2w_mem[c*24+j*3+k];

    for(j=0;j<8;j=j+1)
      for(c=0;c<4;c=c+1) clfw[j][c]=clfw_mem[c*8+j];

    for(c=0;c<4;c=c+1) begin class_total[c]=0; class_correct[c]=0; end
    matches=0; pred_matches=0; err_count=0;

    rst_n=0; start=0; window_valid=0; window_sample=0;
    repeat(5) @(posedge clk);
    rst_n=1;

    for(w=0;w<N;w=w+1) begin
      @(posedge clk); start=1;
      @(posedge clk); start=0;

      for(s=0;s<W;s=s+1) begin
        while(!window_ready) @(posedge clk);
        window_sample=in_mem[w*W+s];
        window_valid=1;
        @(posedge clk);
        window_valid=0;
      end

      wait(done);
      #1;

      for(c=0;c<4;c=c+1) begin
        got=logits_out[c];
        refv=ref_mem[w*4+c];
        if(got==refv) matches=matches+1;
        else if(w<32) $display("LOGIT MISMATCH w=%0d c=%0d got=%0d ref=%0d",w,c,got,refv);
      end

      best=0;
      for(c=1;c<4;c=c+1)
        if(logits_out[c]>logits_out[best]) best=c;
      got_pred=best;
      ref_pred=pred_mem[w];
      ref_label=label_mem[w];
      if(got_pred==ref_pred) pred_matches=pred_matches+1;
      else begin
        err_count=err_count+1;
        if(err_count<=20) $display("PRED MISMATCH w=%0d got=%0d ref=%0d label=%0d",w,got_pred,ref_pred,ref_label);
      end
      class_total[ref_label]=class_total[ref_label]+1;
      if(got_pred==ref_label) class_correct[ref_label]=class_correct[ref_label]+1;

      if((w%64)==63) $display("Progress: %0d/%0d windows",w+1,N);
      @(posedge clk);
    end

    $display("");
    $display("=== CNN E2E RESULT ===");
    $display("Windows: %0d",N);
    $display("Exact logits: %0d/%0d",matches,N*4);
    $display("Prediction/reference: %0d/%0d",pred_matches,N);
    $display("True-label accuracy: %0d/%0d",class_correct[0]+class_correct[1]+class_correct[2]+class_correct[3],N);
    for(c=0;c<4;c=c+1) $display("Class %0d: %0d/%0d",c,class_correct[c],class_total[c]);
    $display("Reference expected accuracy: 458/512 = 89.453125%%");

    if(pred_matches != 458) begin
      $display("ERROR: RTL/reference prediction count is %0d, expected 458",pred_matches);
      $finish(1);
    end
    if(matches != N*4) begin
      $display("ERROR: Exact logits mismatch");
      $finish(1);
    end

    $display("CNN_E2E_TEST_PASSED");
    $finish(0);
  end

  initial begin
    repeat(10000000) @(posedge clk);
    $display("ERROR: CNN simulation timeout");
    $finish(1);
  end
endmodule
