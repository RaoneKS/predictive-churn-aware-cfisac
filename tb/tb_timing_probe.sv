// =============================================================================
// tb_timing_probe.sv — Phase-1: exact latency measurement + signed INT8 checks
// Icarus 12: no task with unpacked-array ports; all inline.
// =============================================================================
`timescale 1ns/1ps

module tb_timing_probe;
  import pkg_accelerator::*;

  localparam real CLK_PERIOD = 5.0;
  logic clk = 0;
  always #(CLK_PERIOD/2.0) clk = ~clk;
  logic rst_n;

  logic        start, abort_req, busy, done_sig;
  layer_desc_t layer_desc_in;
  logic        layer_desc_valid, layer_desc_ready;

  logic [63:0] s_axis_tdata;
  logic [7:0]  s_axis_tkeep;
  logic        s_axis_tvalid, s_axis_tready, s_axis_tlast;

  logic [63:0] m_axis_tdata;
  logic [7:0]  m_axis_tkeep;
  logic        m_axis_tvalid, m_axis_tready, m_axis_tlast;

  logic        wgt_wr_en;
  logic [9:0]  wgt_wr_addr;
  logic [63:0] wgt_wr_data;
  logic        wgt_load_done;

  logic [3:0]  current_state;
  logic        error_flag;
  error_code_t error_code;
  logic        any_overflow;
  logic [31:0] perf_counters [8];

  logic dbg_sram_active_bank, dbg_sram_bank_conflict, dbg_sram_rd_valid;
  logic dbg_act_valid, dbg_act_frame_last;
  logic dbg_quant_valid, dbg_quant_saturated;
  logic dbg_out_backpressure, dbg_wgt_load_start;

  inference_accelerator_top dut (
    .clk(clk),.rst_n(rst_n),.start(start),.abort_req(abort_req),
    .busy(busy),.done(done_sig),
    .layer_desc_in(layer_desc_in),.layer_desc_valid(layer_desc_valid),
    .layer_desc_ready(layer_desc_ready),
    .s_axis_tdata(s_axis_tdata),.s_axis_tkeep(s_axis_tkeep),
    .s_axis_tvalid(s_axis_tvalid),.s_axis_tready(s_axis_tready),
    .s_axis_tlast(s_axis_tlast),
    .m_axis_tdata(m_axis_tdata),.m_axis_tkeep(m_axis_tkeep),
    .m_axis_tvalid(m_axis_tvalid),.m_axis_tready(m_axis_tready),
    .m_axis_tlast(m_axis_tlast),
    .wgt_wr_en(wgt_wr_en),.wgt_wr_addr(wgt_wr_addr),
    .wgt_wr_data(wgt_wr_data),.wgt_load_done(wgt_load_done),
    .current_state(current_state),.error_flag(error_flag),
    .error_code(error_code),.any_overflow(any_overflow),
    .perf_counters(perf_counters),
    .dbg_sram_active_bank(dbg_sram_active_bank),
    .dbg_sram_bank_conflict(dbg_sram_bank_conflict),
    .dbg_sram_rd_valid(dbg_sram_rd_valid),
    .dbg_act_valid(dbg_act_valid),.dbg_act_frame_last(dbg_act_frame_last),
    .dbg_quant_valid(dbg_quant_valid),.dbg_quant_saturated(dbg_quant_saturated),
    .dbg_out_backpressure(dbg_out_backpressure),
    .dbg_wgt_load_start(dbg_wgt_load_start)
  );

  // Cycle counter
  integer gc;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) gc <= 0; else gc <= gc + 1;
  end

  // Timing capture registers
  integer start_cycle, first_valid_cycle, quant_valid_cycle;
  logic was_compute, seen_mvalid, seen_qvalid;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      start_cycle <= 0; was_compute <= 0;
      seen_mvalid <= 0; first_valid_cycle <= 0;
      seen_qvalid <= 0; quant_valid_cycle <= 0;
    end else begin
      if ((current_state==FSM_COMPUTE) && !was_compute) start_cycle <= gc;
      was_compute <= (current_state==FSM_COMPUTE);
      if (m_axis_tvalid && !seen_mvalid) begin
        seen_mvalid <= 1; first_valid_cycle <= gc;
      end
      if (dbg_quant_valid && !seen_qvalid) begin
        seen_qvalid <= 1; quant_valid_cycle <= gc;
      end
    end
  end

  // Test storage (flat arrays, no task ports)
  logic [63:0] wgts[0:7];
  integer i, c, fail_total;
  logic signed [7:0] got_v, exp_v;

  // Helper: wait for state
  task automatic wait_for_state(input logic [3:0] st);
    integer wc;
    begin
      wc = 0;
      while (current_state != st && wc < 500) begin
        @(posedge clk); wc = wc + 1;
      end
      if (current_state != st)
        $display("  WARNING: timed out waiting for state %0d", st);
    end
  endtask

  // Helper: run one inference from reset to output capture
  // Returns m_axis_tdata at time of output
  // Because we can't pass arrays as task ports in iverilog 12,
  // we use module-level 'wgts' and rely on the caller setting them up.
  task automatic do_inference(
    input logic [63:0] act_beat,
    input logic [4:0]  shift,
    input logic signed [7:0] zp,
    input logic relu_en
  );
    integer j;
    begin
      // Full reset
      rst_n = 0; #(CLK_PERIOD*6); rst_n = 1; #(CLK_PERIOD*4);
      // Reset trackers (they reset on negedge rst_n via always_ff)
      @(posedge clk); #1;

      // Start
      start = 1; @(posedge clk); #1; start = 0;

      // Descriptor
      wait_for_state(FSM_LOAD_DESC);
      @(posedge clk); #1;
      layer_desc_in.in_channels  = 16'd8;
      layer_desc_in.out_channels = 16'd8;
      layer_desc_in.spatial_size = 16'd1;
      layer_desc_in.shift_amount = shift;
      layer_desc_in.zero_point   = zp;
      layer_desc_in.relu_enable  = relu_en;
      layer_desc_in.reserved     = 6'd0;
      layer_desc_valid = 1; @(posedge clk); #1; layer_desc_valid = 0;

      // DMA weights (wgts[] already set by caller)
      wait_for_state(FSM_LOAD_WEIGHTS);
      @(posedge clk); #1;
      for (j = 0; j < 8; j = j + 1) begin
        wgt_wr_en = 1; wgt_wr_addr = j[9:0]; wgt_wr_data = wgts[j];
        @(posedge clk); #1;
      end
      wgt_wr_en = 0; wgt_load_done = 1; @(posedge clk); #1; wgt_load_done = 0;

      // Activation beat
      begin : wait_compute
        integer wcc;
        wcc = 0;
        while (current_state != FSM_COMPUTE && current_state != FSM_LOAD_ACT && wcc < 200) begin
          @(posedge clk); wcc = wcc + 1;
        end
      end
      @(posedge clk); #1;
      s_axis_tdata = act_beat; s_axis_tvalid = 1; s_axis_tlast = 1;
      begin : wait_rdy
        integer wr;
        for (wr = 0; wr < 50 && !s_axis_tready; wr = wr + 1)
          @(posedge clk);
      end
      @(posedge clk); #1;
      s_axis_tvalid = 0; s_axis_tlast = 0;

      // Wait for output
      begin : wait_valid
        integer wv;
        for (wv = 0; wv < 1000 && !m_axis_tvalid; wv = wv + 1)
          @(posedge clk);
      end

      // Let the pipeline settle one more cycle for registered output
      @(posedge clk); #1;
    end
  endtask

  initial begin
    rst_n = 0; start = 0; abort_req = 0;
    layer_desc_valid = 0; layer_desc_in = '0;
    s_axis_tdata = '0; s_axis_tkeep = '1;
    s_axis_tvalid = 0; s_axis_tlast = 0;
    m_axis_tready = 1;
    wgt_wr_en = 0; wgt_wr_addr = '0; wgt_wr_data = '0; wgt_load_done = 0;
    fail_total = 0;

    $display("");
    $display("================================================================");
    $display(" Phase-1 Timing Probe: inference_accelerator_top");
    $display(" pkg::TOTAL_LATENCY=%0d  PRIME_CYCLES=%0d  QUANT_LATENCY=%0d",
             TOTAL_LATENCY, 8+7+3, QUANT_LATENCY);
    $display("================================================================");

    // ==================================================================
    // TEST A: Identity 8x8, A=[1..8], shift=0, zp=0, no ReLU
    //   Expected: out[c] = A[c] = c+1
    // ==================================================================
    for (i = 0; i < 8; i = i + 1) begin
      wgts[i] = 64'h0;
      wgts[i][i*8 +: 8] = 8'sd1;
    end
    begin
      logic [63:0] act;
      logic signed [7:0] exp;
      act = 64'h0;
      for (i = 0; i < 8; i = i + 1)
        act[i*8 +: 8] = (i+1); // [1,2,3,4,5,6,7,8]

      do_inference(act, 5'd0, 8'sd0, 1'b0);

      $display("");
      $display("--- TEST A: Identity A=[1..8] shift=0 relu=0 ---");
      $display("  start_cycle         = %0d", start_cycle);
      $display("  quant_valid_cycle   = %0d", quant_valid_cycle);
      $display("  first_valid_cycle   = %0d (m_axis_tvalid)", first_valid_cycle);
      $display("  COMPUTE→m_tvalid    = %0d cycles", first_valid_cycle - start_cycle);
      $display("  PRIME_CYCLES(8+7+3) = %0d", 18);
      $display("  TOTAL_LATENCY(pkg)  = %0d", TOTAL_LATENCY);
      $display("  PRIME+QUANT         = %0d", 18+QUANT_LATENCY);

      $display("  output [%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d]",
               $signed(m_axis_tdata[0+:8]), $signed(m_axis_tdata[8+:8]),
               $signed(m_axis_tdata[16+:8]),$signed(m_axis_tdata[24+:8]),
               $signed(m_axis_tdata[32+:8]),$signed(m_axis_tdata[40+:8]),
               $signed(m_axis_tdata[48+:8]),$signed(m_axis_tdata[56+:8]));
      for (c = 0; c < 8; c = c + 1) begin
        got_v = $signed(m_axis_tdata[c*8+:8]);
        exp_v = $signed(c+1);
        if (got_v !== exp_v) begin
          $display("  MISMATCH col=%0d got=%0d exp=%0d", c, got_v, exp_v);
          fail_total = fail_total + 1;
        end
      end
      if (fail_total == 0) $display("  Test A: PASS");
    end

    // ==================================================================
    // TEST B: Negative signed: W=all-(-1), A=all(+10), relu=0, shift=0
    //   acc = sum_i A[i]*W[i,c] = 10*(-1)*8 = -80 per output
    //   shift=0, relu=0 → out = saturate(-80) = -80
    // ==================================================================
    for (i = 0; i < 8; i = i + 1)
      wgts[i] = 64'hffffffffffffffff; // each byte = 8'hff = -1 signed
    begin
      logic [63:0] act;
      logic signed [7:0] exp;
      act = 64'h0;
      for (i = 0; i < 8; i = i + 1)
        act[i*8 +: 8] = 8'sd10; // +10

      do_inference(act, 5'd0, 8'sd0, 1'b0);

      $display("");
      $display("--- TEST B: W=all(-1) A=all(+10) shift=0 relu=0 → out=-80 ---");
      $display("  COMPUTE→m_tvalid = %0d cycles", first_valid_cycle - start_cycle);
      $display("  output [%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d]",
               $signed(m_axis_tdata[0+:8]),  $signed(m_axis_tdata[8+:8]),
               $signed(m_axis_tdata[16+:8]), $signed(m_axis_tdata[24+:8]),
               $signed(m_axis_tdata[32+:8]), $signed(m_axis_tdata[40+:8]),
               $signed(m_axis_tdata[48+:8]), $signed(m_axis_tdata[56+:8]));
      for (c = 0; c < 8; c = c + 1) begin
        got_v = $signed(m_axis_tdata[c*8+:8]);
        exp_v = -8'sd80;
        if (got_v !== exp_v) begin
          $display("  MISMATCH col=%0d got=%0d exp=%0d", c, got_v, exp_v);
          fail_total = fail_total + 1;
        end
      end
      if (fail_total == 0) $display("  Test B: PASS (signed neg×pos correct)");
    end

    // ==================================================================
    // TEST C: neg×neg = pos: W=all(-1), A=all(-10), relu=1, shift=0
    //   acc = (-10)*(-1)*8 = +80 per output
    //   relu(+80)>>0 = 80
    // ==================================================================
    for (i = 0; i < 8; i = i + 1)
      wgts[i] = 64'hffffffffffffffff;
    begin
      logic [63:0] act;
      act = 64'h0;
      for (i = 0; i < 8; i = i + 1)
        act[i*8 +: 8] = -8'sd10; // -10

      do_inference(act, 5'd0, 8'sd0, 1'b1);

      $display("");
      $display("--- TEST C: W=all(-1) A=all(-10) shift=0 relu=1 → out=+80 ---");
      $display("  COMPUTE→m_tvalid = %0d cycles", first_valid_cycle - start_cycle);
      $display("  output [%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d]",
               $signed(m_axis_tdata[0+:8]),  $signed(m_axis_tdata[8+:8]),
               $signed(m_axis_tdata[16+:8]), $signed(m_axis_tdata[24+:8]),
               $signed(m_axis_tdata[32+:8]), $signed(m_axis_tdata[40+:8]),
               $signed(m_axis_tdata[48+:8]), $signed(m_axis_tdata[56+:8]));
      for (c = 0; c < 8; c = c + 1) begin
        got_v = $signed(m_axis_tdata[c*8+:8]);
        exp_v = 8'sd80;
        if (got_v !== exp_v) begin
          $display("  MISMATCH col=%0d got=%0d exp=%0d", c, got_v, exp_v);
          fail_total = fail_total + 1;
        end
      end
      if (fail_total == 0) $display("  Test C: PASS (neg×neg=pos, ReLU preserves)");
    end

    // ==================================================================
    // TEST D: ReLU clamping: W=identity, A=[-4..-4 all], relu=1
    //   acc[c] = -4 (negative). ReLU → 0. shift=0 → out=0
    // ==================================================================
    for (i = 0; i < 8; i = i + 1) begin
      wgts[i] = 64'h0;
      wgts[i][i*8 +: 8] = 8'sd1;
    end
    begin
      logic [63:0] act;
      act = 64'h0;
      for (i = 0; i < 8; i = i + 1)
        act[i*8 +: 8] = -8'sd4; // -4

      do_inference(act, 5'd0, 8'sd0, 1'b1);

      $display("");
      $display("--- TEST D: W=Identity A=all(-4) relu=1 → out=0 ---");
      $display("  COMPUTE→m_tvalid = %0d cycles", first_valid_cycle - start_cycle);
      $display("  output [%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d]",
               $signed(m_axis_tdata[0+:8]),  $signed(m_axis_tdata[8+:8]),
               $signed(m_axis_tdata[16+:8]), $signed(m_axis_tdata[24+:8]),
               $signed(m_axis_tdata[32+:8]), $signed(m_axis_tdata[40+:8]),
               $signed(m_axis_tdata[48+:8]), $signed(m_axis_tdata[56+:8]));
      for (c = 0; c < 8; c = c + 1) begin
        got_v = $signed(m_axis_tdata[c*8+:8]);
        exp_v = 8'sd0;
        if (got_v !== exp_v) begin
          $display("  MISMATCH col=%0d got=%0d exp=%0d", c, got_v, exp_v);
          fail_total = fail_total + 1;
        end
      end
      if (fail_total == 0) $display("  Test D: PASS (ReLU clamps negatives to 0)");
    end

    // ==================================================================
    // TEST E: Shift=8, relu=1 — matches Conv1/Conv2 requantization
    //   W=identity, A=[64,64,..64], shift=8 → acc=64, >>8=0
    //   A=[128], shift=8 → acc=127 (sat), >>8=0
    //   A=[256] would overflow but 127*1=127, >>8=0 (still 0 here)
    //   Use A=[127]*8, W=identity → acc[c]=127, >>8=0
    // ==================================================================
    for (i = 0; i < 8; i = i + 1) begin
      wgts[i] = 64'h0;
      wgts[i][i*8 +: 8] = 8'sd1;
    end
    begin
      logic [63:0] act;
      act = 64'h0;
      for (i = 0; i < 8; i = i + 1)
        act[i*8 +: 8] = 8'sd127;

      do_inference(act, 5'd8, 8'sd0, 1'b1);

      $display("");
      $display("--- TEST E: W=Identity A=all(127) shift=8 relu=1 → acc=127, >>8=0 ---");
      $display("  COMPUTE→m_tvalid = %0d cycles", first_valid_cycle - start_cycle);
      $display("  output [%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d]",
               $signed(m_axis_tdata[0+:8]),  $signed(m_axis_tdata[8+:8]),
               $signed(m_axis_tdata[16+:8]), $signed(m_axis_tdata[24+:8]),
               $signed(m_axis_tdata[32+:8]), $signed(m_axis_tdata[40+:8]),
               $signed(m_axis_tdata[48+:8]), $signed(m_axis_tdata[56+:8]));
      for (c = 0; c < 8; c = c + 1) begin
        got_v = $signed(m_axis_tdata[c*8+:8]);
        exp_v = 8'sd0; // 127 >>> 8 = 0
        if (got_v !== exp_v) begin
          $display("  MISMATCH col=%0d got=%0d exp=%0d", c, got_v, exp_v);
          fail_total = fail_total + 1;
        end
      end
      if (fail_total == 0) $display("  Test E: PASS (shift=8 correct)");
    end

    $display("");
    $display("================================================================");
    $display(" PHASE-1 FINAL: %s  (fail_total=%0d)",
             (fail_total==0) ? "ALL PASS" : "FAIL", fail_total);
    $display("================================================================");
    $display("");
    $finish;
  end

  initial begin #(CLK_PERIOD*20000); $display("GLOBAL TIMEOUT"); $finish; end
endmodule
