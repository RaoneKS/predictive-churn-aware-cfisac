module alu (
    input [15:0] a,
    input [15:0] b,
    input [3:0] alu_control,
    output reg [15:0] result,
    output zero
);
    assign zero = (result == 16'b0);

    always @(*) begin
        case (alu_control)
            4'b0001: result = b;           // MOV / Pass B
            4'b0011: result = a + b;       // ADD
            4'b0100: result = a - b;       // SUB
            4'b0101: result = a & b;       // AND
            4'b0110: result = a | b;       // OR
            4'b0111: result = a ^ b;       // XOR
            4'b1101: result = a << b[3:0]; // SHL
            4'b1110: result = a >> b[3:0]; // SHR
            default: result = 16'b0;       // NOP / Default
        endcase
    end
endmodule
