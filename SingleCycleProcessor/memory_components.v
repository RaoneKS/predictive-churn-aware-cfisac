module pc_reg (
    input clk,
    input reset,
    input halt,
    input [15:0] next_pc,
    output reg [15:0] pc
);
    always @(posedge clk or posedge reset) begin
        if (reset) pc <= 16'b0;
        else if (!halt) pc <= next_pc;
    end
endmodule

module inst_mem (
    input [15:0] read_addr,
    output [15:0] instruction
);
    reg [15:0] memory [0:255];
    initial $readmemb("instructions.mem", memory); // Load test program
    assign instruction = memory[read_addr[7:0]];
endmodule

module data_mem (
    input clk,
    input mem_read,
    input mem_write,
    input [15:0] address,
    input [15:0] write_data,
    output reg [15:0] read_data
);
    reg [15:0] memory [255:0];
    
    always @(*) begin
        if (mem_read) read_data = memory[address[7:0]];
        else read_data = 16'b0;
    end

    always @(posedge clk) begin
        if (mem_write) memory[address[7:0]] <= write_data;
    end
endmodule
