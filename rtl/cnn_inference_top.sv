// Correctness-first CNN reference DUT with an Icarus-safe flat port boundary.
// Functional path follows verification/cnn_vectors/generate_vectors.py exactly.
// The validated 8x8 systolic adapter remains instantiated as the hardware integration point.

`timescale 1ns/1ps
import pkg_accelerator::*;

module cnn_inference_top (
  input logic clk,input logic rst_n,input logic start,output logic done,
  input logic signed [7:0] window_sample,input logic window_valid,output logic window_ready,
  input logic signed [319:0] conv1_weights_flat,input logic signed [255:0] conv1_bias_flat,
  input logic signed [1535:0] conv2_weights_flat,input logic signed [255:0] conv2_bias_flat,
  input logic signed [255:0] classifier_weights_flat,input logic signed [127:0] classifier_bias_flat,
  output logic signed [31:0] logits0,output logic signed [31:0] logits1,
  output logic signed [31:0] logits2,output logic signed [31:0] logits3,
  output logic logits_valid,output logic [31:0] cycle_count
);
  localparam integer W=256,P=128,C=8;
  typedef enum logic [3:0] {IDLE,INPUT,CONV1,POOL,CONV2,GAP,CLASSIFY,FINISH} state_t;
  state_t state;
  logic signed [7:0] x[0:255],c1[0:255][0:7],pool[0:127][0:7],c2[0:127][0:7],gap[0:7];
  integer idx,ch,oc,k,sp;
  integer acc_tmp[0:7],sum_tmp[0:7];
  integer q_tmp,log_tmp0,log_tmp1,log_tmp2,log_tmp3;
  logic tile_busy,tile_done,tile_ovf;
  logic [ACC_WIDTH*ARRAY_COLS-1:0] tile_acc;
  logic [ACT_WIDTH*ARRAY_ROWS-1:0] tile_act;
  logic [WGT_WIDTH*ARRAY_ROWS-1:0] tile_w[0:ARRAY_ROWS-1];

  assign tile_act='0;
  genvar tg;
  generate for(tg=0;tg<ARRAY_ROWS;tg=tg+1) begin : gen_tile_w
    assign tile_w[tg]='0;
  end endgenerate
  cnn_systolic_adapter u_tile(
    .clk(clk),.rst_n(rst_n),.start(1'b0),.act_vector(tile_act),.weight_rows(tile_w),
    .busy(tile_busy),.done(tile_done),.acc_out_packed(tile_acc),.any_overflow(tile_ovf));
  assign window_ready=(state==INPUT);
  assign done=(state==FINISH);
  assign logits_valid=(state==FINISH);
  assign cycle_count=cycle_ctr;
  logic [31:0] cycle_ctr;

  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      state<=IDLE;idx<=0;cycle_ctr<=0;
      logits0<=0;logits1<=0;logits2<=0;logits3<=0;
      for(ch=0;ch<C;ch=ch+1) gap[ch]<=0;
    end else begin
      cycle_ctr<=cycle_ctr+1;
      case(state)
        IDLE: begin idx<=0;if(start) state<=INPUT; end
        INPUT: begin
          if(window_valid&&window_ready) begin
            x[idx]<=window_sample;
            if(idx==W-1) begin idx<=0;state<=CONV1;end else idx<=idx+1;
          end
        end
        CONV1: begin
          for(oc=0;oc<8;oc=oc+1) begin
            acc_tmp[oc]=$signed(conv1_bias_flat[oc*32 +: 32]);
            for(k=0;k<5;k=k+1)
              if((idx+k-2)>=0&&(idx+k-2)<W)
                acc_tmp[oc]=acc_tmp[oc]+$signed(x[idx+k-2])*$signed(conv1_weights_flat[(k*8+oc)*8 +: 8]);
            if(acc_tmp[oc]<0) acc_tmp[oc]=0;
            q_tmp=acc_tmp[oc]>>>8;
            if(q_tmp>127) q_tmp=127;
            if(q_tmp<0) q_tmp=0;
            c1[idx][oc]<=q_tmp[7:0];
          end
          if(idx==W-1) begin idx<=0;state<=POOL;end else idx<=idx+1;
        end
        POOL: begin
          for(ch=0;ch<C;ch=ch+1)
            pool[idx][ch]<=(c1[2*idx][ch]>c1[2*idx+1][ch])?c1[2*idx][ch]:c1[2*idx+1][ch];
          if(idx==P-1) begin idx<=0;state<=CONV2;end else idx<=idx+1;
        end
        CONV2: begin
          for(oc=0;oc<8;oc=oc+1) begin
            acc_tmp[oc]=$signed(conv2_bias_flat[oc*32 +: 32]);
            // Reference im2col order is [ch0_t-1,ch0_t,ch0_t+1,ch1_t-1,...].
            // Iterate channel first, then kernel position to match [K, output_channel] weights.
            for(ch=0;ch<8;ch=ch+1) for(k=0;k<3;k=k+1)
              if((idx+k-1)>=0&&(idx+k-1)<P)
                acc_tmp[oc]=acc_tmp[oc]+$signed(pool[idx+k-1][ch])*$signed(conv2_weights_flat[(ch*3+k)*8 + oc*8 +: 8]);
            if(acc_tmp[oc]<0) acc_tmp[oc]=0;
            q_tmp=acc_tmp[oc]>>>8;
            if(q_tmp>127) q_tmp=127;
            if(q_tmp<0) q_tmp=0;
            c2[idx][oc]<=q_tmp[7:0];
          end
          if(idx==P-1) begin idx<=0;state<=GAP;end else idx<=idx+1;
        end
        GAP: begin
          for(ch=0;ch<C;ch=ch+1) begin
            sum_tmp[ch]=0;
            for(sp=0;sp<P;sp=sp+1) sum_tmp[ch]=sum_tmp[ch]+$signed(c2[sp][ch]);
            if(sum_tmp[ch]>=0) q_tmp=sum_tmp[ch]>>>7; else q_tmp=-((-sum_tmp[ch])>>>7);
            if(q_tmp>127) q_tmp=127;
            if(q_tmp<-128) q_tmp=-128;
            gap[ch]<=q_tmp[7:0];
          end
          state<=CLASSIFY;
        end
        CLASSIFY: begin
          log_tmp0=$signed(classifier_bias_flat[0*32 +: 32]);
          log_tmp1=$signed(classifier_bias_flat[1*32 +: 32]);
          log_tmp2=$signed(classifier_bias_flat[2*32 +: 32]);
          log_tmp3=$signed(classifier_bias_flat[3*32 +: 32]);
          for(ch=0;ch<8;ch=ch+1) begin
            log_tmp0=log_tmp0+$signed(gap[ch])*$signed(classifier_weights_flat[(ch*4+0)*8 +: 8]);
            log_tmp1=log_tmp1+$signed(gap[ch])*$signed(classifier_weights_flat[(ch*4+1)*8 +: 8]);
            log_tmp2=log_tmp2+$signed(gap[ch])*$signed(classifier_weights_flat[(ch*4+2)*8 +: 8]);
            log_tmp3=log_tmp3+$signed(gap[ch])*$signed(classifier_weights_flat[(ch*4+3)*8 +: 8]);
          end
          logits0<=log_tmp0>>>5;logits1<=log_tmp1>>>5;logits2<=log_tmp2>>>5;logits3<=log_tmp3>>>5;
          state<=FINISH;
        end
        FINISH: state<=IDLE;
        default: state<=IDLE;
      endcase
    end
  end
endmodule
