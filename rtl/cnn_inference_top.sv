// Correctness-first CNN reference DUT.
// Uses manifest dimensions and exact INT8 arithmetic. The systolic tile adapter is
// instantiated for hardware-integration coverage; this top currently uses the
// deterministic scalar path so that the end-to-end reference is independently
// verifiable before replacing each stage with the adapter.

`timescale 1ns/1ps
import pkg_accelerator::*;

module cnn_inference_top (
  input  logic clk, input logic rst_n, input logic start, output logic done,
  input logic signed [7:0] window_sample, input logic window_valid, output logic window_ready,
  input logic signed [7:0] conv1_weights [5][8], input logic signed [31:0] conv1_bias [8],
  input logic signed [7:0] conv2_weights [3][8][8], input logic signed [31:0] conv2_bias [8],
  input logic signed [7:0] classifier_weights [8][4], input logic signed [31:0] classifier_bias [4],
  output logic signed [31:0] logits_out [4], output logic logits_valid, output logic [31:0] cycle_count
);

  localparam integer W=256, P=128, C=8;
  typedef enum logic [3:0] {IDLE,INPUT,CONV1,POOL,CONV2,GAP,CLASSIFY,FINISH} state_t;
  state_t state;

  logic signed [7:0] x[0:255], c1[0:255][0:7], pool[0:127][0:7], c2[0:127][0:7], gap[0:7];
  logic signed [31:0] logits[0:3];
  integer idx, ch, oc, k, sp;
  integer acc_tmp[0:7];
  integer sum_tmp[0:7];
  integer q_tmp;

  // Hardware tile integration point. Functional output remains scalar until the
  // adapter is proven cycle-accurate against this reference.
  logic tile_busy, tile_done, tile_ovf;
  logic [ACC_WIDTH*ARRAY_COLS-1:0] tile_acc;
  logic [ACT_WIDTH*ARRAY_ROWS-1:0] tile_act;
  logic [WGT_WIDTH*ARRAY_ROWS-1:0] tile_w[ARRAY_ROWS];
  assign tile_act='0;
  genvar tg;
  generate for (tg=0;tg<ARRAY_ROWS;tg=tg+1) begin : gen_tile_w
    assign tile_w[tg]='0;
  end endgenerate
  cnn_systolic_adapter u_tile(
    .clk(clk),.rst_n(rst_n),.start(1'b0),.act_vector(tile_act),.weight_rows(tile_w),
    .busy(tile_busy),.done(tile_done),.acc_out_packed(tile_acc),.any_overflow(tile_ovf));

  assign window_ready = (state==INPUT);
  assign done = (state==FINISH);
  assign logits_valid = (state==FINISH);
  assign cycle_count = cycle_ctr;
  always_comb begin
    logits_out[0]=logits[0]; logits_out[1]=logits[1];
    logits_out[2]=logits[2]; logits_out[3]=logits[3];
  end

  logic [31:0] cycle_ctr;
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      state<=IDLE; idx<=0; cycle_ctr<=0;
      for(ch=0;ch<C;ch=ch+1) gap[ch]<='0;
      for(oc=0;oc<4;oc=oc+1) logits[oc]<='0;
    end else begin
      cycle_ctr<=cycle_ctr+1;
      case(state)
        IDLE: begin
          idx<=0;
          if(start) state<=INPUT;
        end
        INPUT: begin
          if(window_valid && window_ready) begin
            x[idx]<=window_sample;
            if(idx==255) begin idx<=0; state<=CONV1; end
            else idx<=idx+1;
          end
        end
        CONV1: begin
          for(oc=0;oc<8;oc=oc+1) begin
            acc_tmp[oc]=conv1_bias[oc];
            for(k=0;k<5;k=k+1) begin
              if((idx+k)>=2 && (idx+k)<=257) begin
                if((idx+k-2)>=0 && (idx+k-2)<256)
                  acc_tmp[oc]=acc_tmp[oc]+$signed(x[idx+k-2])*$signed(conv1_weights[k][oc]);
              end
            end
            if(acc_tmp[oc]<0) acc_tmp[oc]=0;
            q_tmp=acc_tmp[oc]>>>8;
            if(q_tmp>127) q_tmp=127;
            if(q_tmp<0) q_tmp=0;
            c1[idx][oc]<=q_tmp[7:0];
          end
          if(idx==255) begin idx<=0; state<=POOL; end else idx<=idx+1;
        end
        POOL: begin
          for(ch=0;ch<8;ch=ch+1)
            pool[idx][ch] <= (c1[2*idx][ch] > c1[2*idx+1][ch]) ? c1[2*idx][ch] : c1[2*idx+1][ch];
          if(idx==127) begin idx<=0; state<=CONV2; end else idx<=idx+1;
        end
        CONV2: begin
          for(oc=0;oc<8;oc=oc+1) begin
            acc_tmp[oc]=conv2_bias[oc];
            for(k=0;k<3;k=k+1)
              for(ch=0;ch<8;ch=ch+1)
                if((idx+k)>=1 && (idx+k)<=127) begin
                  if((idx+k-1)>=0 && (idx+k-1)<128)
                    acc_tmp[oc]=acc_tmp[oc]+$signed(pool[idx+k-1][ch])*$signed(conv2_weights[k][ch][oc]);
                end
            if(acc_tmp[oc]<0) acc_tmp[oc]=0;
            q_tmp=acc_tmp[oc]>>>8;
            if(q_tmp>127) q_tmp=127;
            if(q_tmp<0) q_tmp=0;
            c2[idx][oc]<=q_tmp[7:0];
          end
          if(idx==127) begin idx<=0; state<=GAP; end else idx<=idx+1;
        end
        GAP: begin
          for(ch=0;ch<8;ch=ch+1) begin
            sum_tmp[ch]=0;
            for(sp=0;sp<128;sp=sp+1) sum_tmp[ch]=sum_tmp[ch]+$signed(c2[sp][ch]);
            if(sum_tmp[ch]>=0) q_tmp=sum_tmp[ch]>>>7;
            else q_tmp=-((-sum_tmp[ch])>>>7);
            if(q_tmp>127) q_tmp=127;
            if(q_tmp< -128) q_tmp=-128;
            gap[ch]<=q_tmp[7:0];
          end
          state<=CLASSIFY;
        end
        CLASSIFY: begin
          for(oc=0;oc<4;oc=oc+1) begin
            acc_tmp[oc]=classifier_bias[oc];
            for(ch=0;ch<8;ch=ch+1)
              acc_tmp[oc]=acc_tmp[oc]+$signed(gap[ch])*$signed(classifier_weights[ch][oc]);
            logits[oc]<=acc_tmp[oc]>>>5;
          end
          state<=FINISH;
        end
        FINISH: state<=IDLE;
        default: state<=IDLE;
      endcase
    end
  end
endmodule
