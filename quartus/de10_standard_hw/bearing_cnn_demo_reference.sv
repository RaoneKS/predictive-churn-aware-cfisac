// Fixed-weight FPGA deployment engine for the exported Phase-3 CWRU CNN.
// It deliberately leaves the validated generic accelerator RTL unchanged.
module bearing_cnn_demo (
  input logic clk, input logic rst_n, input logic start, input logic select_outer,
  output logic busy, output logic done, output logic [1:0] predicted_class,
  output logic [3:0] state, output logic [31:0] logits_out
);
  localparam IDLE=0, C1=1, POOL=2, C2=3, GAP=4, FC=5, PICK=6, DONE=7;
  logic [3:0] st;
  logic select_latched;
  logic [8:0] pos; logic [4:0] k; logic [3:0] oc;
  logic signed [31:0] acc, gap_sum;
  logic signed [7:0] input_rom [0:511];
  logic signed [7:0] c1_w [0:63], c2_w [0:191], fc_w [0:63];
  logic signed [31:0] c1_b [0:7], c2_b [0:7], fc_b [0:3];
  logic signed [7:0] c1_mem [0:2047], pool_mem [0:1023], c2_mem [0:1023];
  logic signed [7:0] gap_mem [0:7], logits [0:7];
  logic signed [7:0] sample, weight;
  logic signed [31:0] bias, product;
  integer tap, chan, address;

  initial begin
    $readmemh("model_data/demo_inputs.hex", input_rom);
    $readmemh("model_data/conv1_weights.hex", c1_w);
    $readmemh("model_data/conv1_bias.hex", c1_b);
    $readmemh("model_data/conv2_weights.hex", c2_w);
    $readmemh("model_data/conv2_bias.hex", c2_b);
    $readmemh("model_data/classifier_weights.hex", fc_w);
    $readmemh("model_data/classifier_bias.hex", fc_b);
  end

  function automatic logic signed [7:0] quantize(input logic signed [31:0] value,
                                                   input logic [4:0] shift,
                                                   input logic relu);
    logic signed [31:0] tmp;
    begin
      tmp = relu && value[31] ? 32'sd0 : value;
      tmp = tmp >>> shift;
      if (tmp > 32'sd127) quantize = 8'sd127;
      else if (tmp < -32'sd128) quantize = -8'sd128;
      else quantize = tmp[7:0];
    end
  endfunction

  always_comb begin
    sample='0; weight='0; bias='0; tap=0; chan=0; address=0;
    case (st)
      C1: begin
        tap=k;
        if ((pos + tap) >= 2 && (pos + tap) < 258)
          sample=input_rom[(select_latched ? 256 : 0) + pos + tap - 2];
        weight=c1_w[tap*8+oc]; bias=c1_b[oc];
      end
      C2: begin
        chan=k/3; tap=k%3;
        if ((pos + tap) >= 1 && (pos + tap) < 129)
          sample=pool_mem[(pos+tap-1)*8+chan];
        weight=c2_w[k*8+oc]; bias=c2_b[oc];
      end
      FC: begin
        sample=gap_mem[k]; weight=fc_w[k*8+oc];
        if (oc<4) bias=fc_b[oc];
      end
      default: ;
    endcase
    product=sample*weight;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st<=IDLE; pos<='0; k<='0; oc<='0; acc<='0; gap_sum<='0;
      predicted_class<='0; select_latched<=1'b0;
    end else begin
      case (st)
        IDLE: if (start) begin
          st<=C1; pos<=0; k<=0; oc<=0; acc<=0; select_latched<=select_outer;
        end
        C1: if (k==4) begin
          c1_mem[pos*8+oc] <= quantize(acc+product+bias,5'd8,1'b1);
          acc<=0; k<=0;
          if (oc==7) begin oc<=0; if (pos==255) begin pos<=0; st<=POOL; end else pos<=pos+1'b1; end
          else oc<=oc+1'b1;
        end else begin acc<=acc+product; k<=k+1'b1; end
        POOL: begin
          pool_mem[pos*8+oc] <= (c1_mem[(pos*2)*8+oc] > c1_mem[(pos*2+1)*8+oc]) ?
                                 c1_mem[(pos*2)*8+oc] : c1_mem[(pos*2+1)*8+oc];
          if (oc==7) begin oc<=0; if (pos==127) begin pos<=0; k<=0; st<=C2; end else pos<=pos+1'b1; end
          else oc<=oc+1'b1;
        end
        C2: if (k==23) begin
          c2_mem[pos*8+oc] <= quantize(acc+product+bias,5'd8,1'b1);
          acc<=0; k<=0;
          if (oc==7) begin oc<=0; if (pos==127) begin pos<=0; st<=GAP; end else pos<=pos+1'b1; end
          else oc<=oc+1'b1;
        end else begin acc<=acc+product; k<=k+1'b1; end
        GAP: begin
          if (pos==127) begin
            gap_mem[oc] <= (gap_sum+c2_mem[pos*8+oc]) >>> 7;
            gap_sum<=0; pos<=0;
            if (oc==7) begin oc<=0; k<=0; st<=FC; end else oc<=oc+1'b1;
          end else begin gap_sum<=gap_sum+c2_mem[pos*8+oc]; pos<=pos+1'b1; end
        end
        FC: if (k==7) begin
          logits[oc] <= quantize(acc+product+bias,5'd5,1'b0);
          acc<=0; k<=0;
          if (oc==7) st<=PICK; else oc<=oc+1'b1;
        end else begin acc<=acc+product; k<=k+1'b1; end
        PICK: begin
          if (logits[1]>logits[0] && logits[1]>=logits[2] && logits[1]>=logits[3]) predicted_class<=2'd1;
          else if (logits[2]>logits[0] && logits[2]>logits[1] && logits[2]>=logits[3]) predicted_class<=2'd2;
          else if (logits[3]>logits[0] && logits[3]>logits[1] && logits[3]>logits[2]) predicted_class<=2'd3;
          else predicted_class<=2'd0;
          st<=DONE;
        end
        DONE: if (!start) st<=IDLE;
        default: st<=IDLE;
      endcase
    end
  end
  assign busy=(st!=IDLE && st!=DONE);
  assign done=(st==DONE);
  assign state=st;
  assign logits_out={logits[3],logits[2],logits[1],logits[0]};
endmodule
