// ---------------------------------------------------------------------------
// tb_systolic_sv.sv  -  SystemVerilog constrained-random testbench for
//                       systolic_array, with functional coverage tracking.
//
// WHY THIS EXISTS (and how it differs from tb_systolic.v):
//
//   The original testbench generates operands with `$random(seed) % 16`, so
//   every value it has ever driven lies in [-15, +15]. On a signed 8-bit
//   datapath that leaves the entire outer range -- including the int8
//   boundaries -128 and +127 -- completely unexercised. Sign-extension and
//   saturation bugs live exactly there.
//
//   This testbench:
//     * drives directed corner cases first (int8 min/max, zeros, sign mixes),
//     * then constrained-random operands across the FULL int8 range,
//     * tracks functional coverage bins and refuses to pass if any bin is
//       empty, so "tests passed" cannot hide "we never tried that".
//
// TOOLCHAIN NOTE:
//   Icarus Verilog 12.0 does not implement SystemVerilog classes, covergroups
//   or UVM. Coverage here is therefore tracked explicitly in arrays rather
//   than with `covergroup`. Everything below compiles and runs under:
//
//     iverilog -g2012 -o sim/tb_sv rtl/pe.v rtl/systolic_array.v tb/tb_systolic_sv.sv
//     vvp sim/tb_sv
//
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps
`default_nettype none

module tb_systolic_sv;

    // ---------------------------------------------------------------- params
    localparam int N          = 4;
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;

    localparam int DATA_MIN   = -(1 << (DATA_WIDTH-1));      // -128
    localparam int DATA_MAX   =  (1 << (DATA_WIDTH-1)) - 1;  // +127

    localparam int NUM_DIRECTED = 6;
    localparam int NUM_RANDOM   = 200;

    // ---------------------------------------------------------------- signals
    logic                        clk, rst_n, en, clear;
    logic [N*DATA_WIDTH-1:0]     a_in_flat, b_in_flat;
    wire  [N*N*ACC_WIDTH-1:0]    c_out_flat;

    logic signed [DATA_WIDTH-1:0] A      [N][N];
    logic signed [DATA_WIDTH-1:0] B      [N][N];
    logic signed [ACC_WIDTH-1:0]  C_gold [N][N];

    int total_errors;
    int tests_run;

    // ------------------------------------------------------- coverage model
    // Value bins for every operand element driven into the DUT.
    typedef enum int {
        BIN_NEG_MAX,    // exactly -128
        BIN_NEG_LARGE,  // -127 .. -64
        BIN_NEG_SMALL,  //  -63 ..  -1
        BIN_ZERO,       //    0
        BIN_POS_SMALL,  //   +1 .. +63
        BIN_POS_LARGE,  //  +64 .. +126
        BIN_POS_MAX,    // exactly +127
        NUM_VAL_BINS
    } val_bin_e;

    // Sign-pair bins for the A*B products actually multiplied.
    typedef enum int {
        SP_NEG_NEG, SP_NEG_POS, SP_POS_NEG, SP_POS_POS, SP_ZERO,
        NUM_SIGN_BINS
    } sign_bin_e;

    int cov_val  [NUM_VAL_BINS];
    int cov_sign [NUM_SIGN_BINS];

    function automatic val_bin_e classify_val(input int v);
        if (v == DATA_MIN)          return BIN_NEG_MAX;
        else if (v <= -64)          return BIN_NEG_LARGE;
        else if (v <   0)           return BIN_NEG_SMALL;
        else if (v ==  0)           return BIN_ZERO;
        else if (v <= 63)           return BIN_POS_SMALL;
        else if (v <  DATA_MAX)     return BIN_POS_LARGE;
        else                        return BIN_POS_MAX;
    endfunction

    function automatic sign_bin_e classify_pair(input int x, input int y);
        if (x == 0 || y == 0)       return SP_ZERO;
        else if (x < 0 && y < 0)    return SP_NEG_NEG;
        else if (x < 0 && y > 0)    return SP_NEG_POS;
        else if (x > 0 && y < 0)    return SP_POS_NEG;
        else                        return SP_POS_POS;
    endfunction

    // Sample coverage for the operand set currently in A / B.
    task automatic sample_coverage();
        int i, j, k;
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++) begin
                cov_val[classify_val(int'(A[i][j]))]++;
                cov_val[classify_val(int'(B[i][j]))]++;
            end
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                for (k = 0; k < N; k++)
                    cov_sign[classify_pair(int'(A[i][k]), int'(B[k][j]))]++;
    endtask

    // ---------------------------------------------------------------- clock
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // ------------------------------------------------------------------ DUT
    systolic_array #(
        .N(N), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n), .en(en), .clear(clear),
        .a_in_flat(a_in_flat), .b_in_flat(b_in_flat),
        .c_out_flat(c_out_flat)
    );

    // ------------------------------------------------------- golden model
    task automatic compute_golden();
        int i, j, k;
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++) begin
                C_gold[i][j] = '0;
                for (k = 0; k < N; k++)
                    C_gold[i][j] += ACC_WIDTH'(A[i][k]) * ACC_WIDTH'(B[k][j]);
            end
    endtask

    // ------------------------------------------------------ stimulus: fill
    // Constrained-random across the FULL int8 range, with deliberate weight
    // on the boundaries -- uniform random almost never produces -128/+127.
    task automatic fill_random();
        int i, j, pick;
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++) begin
                pick = $urandom_range(0, 99);
                if      (pick < 8)  A[i][j] = DATA_MIN[DATA_WIDTH-1:0];
                else if (pick < 16) A[i][j] = DATA_MAX[DATA_WIDTH-1:0];
                else if (pick < 24) A[i][j] = '0;
                else                A[i][j] = DATA_WIDTH'($urandom_range(0, 255));

                pick = $urandom_range(0, 99);
                if      (pick < 8)  B[i][j] = DATA_MIN[DATA_WIDTH-1:0];
                else if (pick < 16) B[i][j] = DATA_MAX[DATA_WIDTH-1:0];
                else if (pick < 24) B[i][j] = '0;
                else                B[i][j] = DATA_WIDTH'($urandom_range(0, 255));
            end
    endtask

    task automatic fill_const(input int va, input int vb);
        int i, j;
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++) begin
                A[i][j] = DATA_WIDTH'(va);
                B[i][j] = DATA_WIDTH'(vb);
            end
    endtask

    // Identity in A, ramp in B -- catches row/column transposition bugs that
    // symmetric random data can mask entirely.
    task automatic fill_identity_ramp();
        int i, j;
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++) begin
                A[i][j] = (i == j) ? DATA_WIDTH'(1) : DATA_WIDTH'(0);
                B[i][j] = DATA_WIDTH'(i*N + j + 1);
            end
    endtask

    // --------------------------------------------------------- drive + check
    task automatic drive_cycle(input int cyc);
        int i, j, idx;
        a_in_flat = '0;
        b_in_flat = '0;
        for (i = 0; i < N; i++) begin
            idx = cyc - i;
            if (idx >= 0 && idx < N)
                a_in_flat[i*DATA_WIDTH +: DATA_WIDTH] = A[i][idx];
        end
        for (j = 0; j < N; j++) begin
            idx = cyc - j;
            if (idx >= 0 && idx < N)
                b_in_flat[j*DATA_WIDTH +: DATA_WIDTH] = B[idx][j];
        end
    endtask

    task automatic run_case(input int tnum, input string tag);
        int i, j, t, errors;
        logic signed [ACC_WIDTH-1:0] c_hw;

        errors = 0;
        compute_golden();
        sample_coverage();

        rst_n = 1'b0; en = 1'b0; clear = 1'b0;
        a_in_flat = '0; b_in_flat = '0;
        @(negedge clk); @(negedge clk);
        rst_n = 1'b1; en = 1'b1;

        for (t = 0; t < 4*N; t++) begin
            @(negedge clk);
            drive_cycle(t);
        end
        repeat (4) @(negedge clk);

        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++) begin
                c_hw = c_out_flat[(i*N + j)*ACC_WIDTH +: ACC_WIDTH];
                if (c_hw !== C_gold[i][j]) begin
                    errors++;
                    $display("  MISMATCH [%0s test %0d] C[%0d][%0d]: hw=%0d gold=%0d",
                             tag, tnum, i, j, c_hw, C_gold[i][j]);
                end
            end

        tests_run++;
        if (errors != 0) begin
            total_errors += errors;
            $display("  test %0d (%0s): FAIL - %0d mismatches", tnum, tag, errors);
        end
    endtask

    // ------------------------------------------------------ coverage report
    function automatic int report_coverage();
        int holes;
        val_bin_e  vb;
        sign_bin_e sb;
        string     vnames [NUM_VAL_BINS];
        string     snames [NUM_SIGN_BINS];

        vnames[BIN_NEG_MAX]   = "operand == -128 (int8 min)";
        vnames[BIN_NEG_LARGE] = "operand in [-127,-64]";
        vnames[BIN_NEG_SMALL] = "operand in [-63,-1]";
        vnames[BIN_ZERO]      = "operand == 0";
        vnames[BIN_POS_SMALL] = "operand in [+1,+63]";
        vnames[BIN_POS_LARGE] = "operand in [+64,+126]";
        vnames[BIN_POS_MAX]   = "operand == +127 (int8 max)";

        snames[SP_NEG_NEG] = "product neg x neg";
        snames[SP_NEG_POS] = "product neg x pos";
        snames[SP_POS_NEG] = "product pos x neg";
        snames[SP_POS_POS] = "product pos x pos";
        snames[SP_ZERO]    = "product with a zero operand";

        holes = 0;
        $display("");
        $display("---- FUNCTIONAL COVERAGE : OPERAND VALUE BINS ----");
        for (int b = 0; b < NUM_VAL_BINS; b++) begin
            vb = val_bin_e'(b);
            $display("  %-30s hits=%0d%0s", vnames[b], cov_val[b],
                     (cov_val[b] == 0) ? "   <== UNCOVERED" : "");
            if (cov_val[b] == 0) holes++;
        end

        $display("---- FUNCTIONAL COVERAGE : PRODUCT SIGN BINS ----");
        for (int b = 0; b < NUM_SIGN_BINS; b++) begin
            sb = sign_bin_e'(b);
            $display("  %-30s hits=%0d%0s", snames[b], cov_sign[b],
                     (cov_sign[b] == 0) ? "   <== UNCOVERED" : "");
            if (cov_sign[b] == 0) holes++;
        end
        return holes;
    endfunction

    // ------------------------------------------------------------------ main
    initial begin
        int holes;

        $dumpfile("sim/dump_sv.vcd");
        $dumpvars(0, tb_systolic_sv);

        total_errors = 0;
        tests_run    = 0;
        foreach (cov_val[i])  cov_val[i]  = 0;
        foreach (cov_sign[i]) cov_sign[i] = 0;

        $display("=== systolic_array SV constrained-random check : %0dx%0d, DW=%0d ===",
                 N, N, DATA_WIDTH);
        $display("    int8 range under test: %0d .. %0d", DATA_MIN, DATA_MAX);

        // ---- directed corner cases -------------------------------------
        $display("--- directed corner cases ---");
        fill_const(DATA_MAX, DATA_MAX);  run_case(1, "max*max");
        fill_const(DATA_MIN, DATA_MIN);  run_case(2, "min*min");
        fill_const(DATA_MIN, DATA_MAX);  run_case(3, "min*max");
        fill_const(0, DATA_MAX);         run_case(4, "zero*max");
        fill_const(-1, DATA_MIN);        run_case(5, "neg1*min");
        fill_identity_ramp();            run_case(6, "identity*ramp");

        // ---- constrained random ----------------------------------------
        $display("--- constrained-random (%0d cases, full int8 range) ---", NUM_RANDOM);
        for (int t = 1; t <= NUM_RANDOM; t++) begin
            fill_random();
            run_case(NUM_DIRECTED + t, "random");
        end

        // ---- results ----------------------------------------------------
        holes = report_coverage();

        $display("");
        $display("================================================");
        $display("  cases run      : %0d", tests_run);
        $display("  mismatches     : %0d", total_errors);
        $display("  coverage holes : %0d", holes);
        $display("================================================");

        // A run only passes if it is BOTH correct and complete. Passing with
        // an empty bin would mean "we never tried that", reported as success.
        if (total_errors != 0) begin
            $display("RESULT: FAIL - functional mismatches detected");
            $fatal;
        end
        else if (holes != 0) begin
            $display("RESULT: FAIL - coverage holes remain");
            $fatal;
        end
        else begin
            $display("RESULT: PASS - all cases bit-exact, all coverage bins hit");
        end

        $finish;
    end

endmodule

`default_nettype wire
