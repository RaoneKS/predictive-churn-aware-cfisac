`timescale 1ns / 1ps

module processor_tb;
    reg clk;
    reg reset;

    // Instantiate Top Module
    processor_top UUT (
        .clk(clk),
        .reset(reset)
    );

    // Clock Generation (10ns period)
    always #5 clk = ~clk;

    initial begin
        // GTKWave Waveforms Setup
        $dumpfile("waveform.vcd");
        $dumpvars(0, processor_tb);

        // Initialize Inputs
        clk = 0;
        reset = 1;

        // Terminal Log Header
        $display("---------------------------------------------------------------");
        $display("Time | PC   | Opcode | ALU Res | Reg[0] | Reg[1] | Reg[2] | Reg[3]");
        $display("---------------------------------------------------------------");
        $monitor("%4d | %4h |  %b  |  %4h   |  %4h  |  %4h  |  %4h  |  %4h", 
                 $time, UUT.pc, UUT.opcode, UUT.alu_result, 
                 UUT.REG.registers[0], UUT.REG.registers[1], 
                 UUT.REG.registers[2], UUT.REG.registers[3]);

        // Release reset
        #10 reset = 0;

        // Run until HALT or safety timeout
        fork
            begin
                wait(UUT.halt == 1);
                #10;
                $display("---------------------------------------------------------------");
                $display("Processor Halted Successfully. Test Complete.");
                $finish;
            end
            begin
                #500;
                $display("Simulation Timeout.");
                $finish;
            end
        join
    end
endmodule
