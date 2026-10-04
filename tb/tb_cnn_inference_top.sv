`timescale 1ns/1ps
module tb_cnn_inference_top;
  localparam integer N=512,W=256;
  logic clk,rst_n,start,done,window_valid,window_ready,logits_valid;
  logic signed [7:0] window_sample;
  logic signed [31:0] logits0,logits1,logits2,logits3;
  logic [31:0] cycle_count;
  logic signed [7:0] c1w_mem[0:39]; logic signed [31:0] c1b_mem[0:7];
  logic signed [7:0] c2w_mem[0:191]; logic signed [31:0] c2b_mem[0:7];
  logic signed [7:0] clfw_mem[0:31]; logic signed [31:0] clfb_mem[0:3];
  logic signed [7:0] in_mem[0:N*W-1]; logic signed [31:0] ref_mem[0:N*4-1];
  logic [7:0] pred_mem[0:N-1]; logic [7:0] label_mem[0:N-1];
  logic signed [319:0] c1w_flat; logic signed [255:0] c1b_flat;
  logic signed [1535:0] c2w_flat; logic signed [255:0] c2b_flat;
  logic signed [255:0] clfw_flat; logic signed [127:0] clfb_flat;
  integer i,j,k,w,s,c,matches,pred_matches,err_count,got_pred,best,got,refv,ref_pred,ref_label;
  integer class_total0,class_total1,class_total2,class_total3;
  integer class_correct0,class_correct1,class_correct2,class_correct3;
  integer repeat0,repeat1,repeat2,repeat3;

  cnn_inference_top dut(
    .clk(clk),.rst_n(rst_n),.start(start),.done(done),
    .window_sample(window_sample),.window_valid(window_valid),.window_ready(window_ready),
    .conv1_weights_flat(c1w_flat),.conv1_bias_flat(c1b_flat),
    .conv2_weights_flat(c2w_flat),.conv2_bias_flat(c2b_flat),
    .classifier_weights_flat(clfw_flat),.classifier_bias_flat(clfb_flat),
    .logits0(logits0),.logits1(logits1),.logits2(logits2),.logits3(logits3),
    .logits_valid(logits_valid),.cycle_count(cycle_count));

  
    genvar g;
    generate
      for(g=0;g<8;g=g+1) begin : GEN_C1B
        assign c1b_flat[g*32 +: 32] = c1b_mem[g];
        assign c2b_flat[g*32 +: 32] = c2b_mem[g];
      end
      for(g=0;g<4;g=g+1) begin : GEN_CLFB
        assign clfb_flat[g*32 +: 32] = clfb_mem[g];
      end
    endgenerate
  
    genvar wk, wc, ck, cc, cj;
    generate
      for(wk=0;wk<5;wk=wk+1) begin : GEN_C1W_K
        for(wc=0;wc<8;wc=wc+1) begin : GEN_C1W_C
          assign c1w_flat[(wk*8+wc)*8 +: 8] = c1w_mem[wc*5+wk];
        end
      end
      for(ck=0;ck<3;ck=ck+1) begin : GEN_C2W_K
        for(cj=0;cj<8;cj=cj+1) begin : GEN_C2W_IC
          for(cc=0;cc<8;cc=cc+1) begin : GEN_C2W_OC
            assign c2w_flat[(ck*64+cj*8+cc)*8 +: 8] = c2w_mem[cc*24+cj*3+ck];
          end
        end
      end
      for(cj=0;cj<8;cj=cj+1) begin : GEN_CLF_J
        for(cc=0;cc<4;cc=cc+1) begin : GEN_CLF_C
          assign clfw_flat[(cj*4+cc)*8 +: 8] = clfw_mem[cc*8+cj];
        end
      end
    endgenerate
  
  initial begin clk=0;forever #5 clk=~clk;end

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


    class_total0=0;class_total1=0;class_total2=0;class_total3=0;
    class_correct0=0;class_correct1=0;class_correct2=0;class_correct3=0;
    matches=0;pred_matches=0;err_count=0;
    rst_n=0;start=0;window_valid=0;window_sample=0;
    repeat(5) @(posedge clk);rst_n=1;

    for(w=0;w<N;w=w+1) begin
      @(posedge clk);start=1;@(posedge clk);start=0;
      for(s=0;s<W;s=s+1) begin
        while(!window_ready) @(posedge clk);
        window_sample=in_mem[w*W+s];window_valid=1;@(posedge clk);window_valid=0;
      end
      wait(done);#1;

      got=logits0;refv=ref_mem[w*4+0];if(got==refv) matches=matches+1;else if(w<32) $display("LOGIT MISMATCH w=%0d c=0 got=%0d ref=%0d",w,got,refv);
      got=logits1;refv=ref_mem[w*4+1];if(got==refv) matches=matches+1;else if(w<32) $display("LOGIT MISMATCH w=%0d c=1 got=%0d ref=%0d",w,got,refv);
      got=logits2;refv=ref_mem[w*4+2];if(got==refv) matches=matches+1;else if(w<32) $display("LOGIT MISMATCH w=%0d c=2 got=%0d ref=%0d",w,got,refv);
      got=logits3;refv=ref_mem[w*4+3];if(got==refv) matches=matches+1;else if(w<32) $display("LOGIT MISMATCH w=%0d c=3 got=%0d ref=%0d",w,got,refv);

      best=0;if(logits1>logits0) best=1;if(logits2>(best==0?logits0:logits1)) best=2;if(logits3>(best==0?logits0:(best==1?logits1:logits2))) best=3;
      got_pred=best;ref_pred=pred_mem[w];ref_label=label_mem[w];
      if(got_pred==ref_pred) pred_matches=pred_matches+1;else begin err_count=err_count+1;if(err_count<=20) $display("PRED MISMATCH w=%0d got=%0d ref=%0d label=%0d",w,got_pred,ref_pred,ref_label);end
      case(ref_label)
        0: begin class_total0=class_total0+1;if(got_pred==ref_label) class_correct0=class_correct0+1;end
        1: begin class_total1=class_total1+1;if(got_pred==ref_label) class_correct1=class_correct1+1;end
        2: begin class_total2=class_total2+1;if(got_pred==ref_label) class_correct2=class_correct2+1;end
        3: begin class_total3=class_total3+1;if(got_pred==ref_label) class_correct3=class_correct3+1;end
      endcase
      if((w%64)==63) $display("Progress: %0d/%0d windows",w+1,N);
      @(posedge clk);
    end

    // Explicit back-to-back normal -> outer without reset.
    @(posedge clk);start=1;@(posedge clk);start=0;
    for(s=0;s<W;s=s+1) begin
      while(!window_ready) @(posedge clk);
      window_sample=in_mem[s];window_valid=1;@(posedge clk);window_valid=0;
    end
    wait(done);#1;
    if(logits0!==ref_mem[0]) begin $display("ERROR: back-to-back normal mismatch c0");$finish(1);end
    if(logits1!==ref_mem[1]) begin $display("ERROR: back-to-back normal mismatch c1");$finish(1);end
    if(logits2!==ref_mem[2]) begin $display("ERROR: back-to-back normal mismatch c2");$finish(1);end
    if(logits3!==ref_mem[3]) begin $display("ERROR: back-to-back normal mismatch c3");$finish(1);end

    @(posedge clk);start=1;@(posedge clk);start=0;
    for(s=0;s<W;s=s+1) begin
      while(!window_ready) @(posedge clk);
      window_sample=in_mem[384*W+s];window_valid=1;@(posedge clk);window_valid=0;
    end
    wait(done);#1;
    if(logits0!==ref_mem[384*4+0]) begin $display("ERROR: back-to-back outer mismatch c0");$finish(1);end
    if(logits1!==ref_mem[384*4+1]) begin $display("ERROR: back-to-back outer mismatch c1");$finish(1);end
    if(logits2!==ref_mem[384*4+2]) begin $display("ERROR: back-to-back outer mismatch c2");$finish(1);end
    if(logits3!==ref_mem[384*4+3]) begin $display("ERROR: back-to-back outer mismatch c3");$finish(1);end

    // Explicit same-window repeat without reset.
    @(posedge clk);start=1;@(posedge clk);start=0;
    for(s=0;s<W;s=s+1) begin
      while(!window_ready) @(posedge clk);
      window_sample=in_mem[s];window_valid=1;@(posedge clk);window_valid=0;
    end
    wait(done);#1;
    repeat0=logits0;repeat1=logits1;repeat2=logits2;repeat3=logits3;

    @(posedge clk);start=1;@(posedge clk);start=0;
    for(s=0;s<W;s=s+1) begin
      while(!window_ready) @(posedge clk);
      window_sample=in_mem[s];window_valid=1;@(posedge clk);window_valid=0;
    end
    wait(done);#1;
    if(logits0!==repeat0 || logits1!==repeat1 || logits2!==repeat2 || logits3!==repeat3) begin
      $display("ERROR: same-window repeat mismatch");
      $finish(1);
    end
    $display("BACK_TO_BACK_TEST_PASSED");
    $display("SAME_WINDOW_REPEAT_PASSED");

    $display("");$display("=== CNN E2E RESULT ===");$display("Windows: %0d",N);
    $display("Exact logits: %0d/%0d",matches,N*4);
    $display("Prediction/reference: %0d/%0d",pred_matches,N);
    $display("True-label accuracy: %0d/%0d",class_correct0+class_correct1+class_correct2+class_correct3,N);
    $display("Class 0: %0d/%0d",class_correct0,class_total0);$display("Class 1: %0d/%0d",class_correct1,class_total1);
    $display("Class 2: %0d/%0d",class_correct2,class_total2);$display("Class 3: %0d/%0d",class_correct3,class_total3);
    $display("Reference expected accuracy: 458/512 = 89.453125%%");
    if(pred_matches!=N) begin $display("ERROR: RTL/reference prediction count is %0d, expected %0d",pred_matches,N);$finish(1);end
    if(matches!=N*4) begin $display("ERROR: Exact logits mismatch");$finish(1);end
    $display("CNN_E2E_TEST_PASSED");$finish(0);
  end

  initial begin repeat(10000000) @(posedge clk);$display("ERROR: CNN simulation timeout");$finish(1);end
endmodule
