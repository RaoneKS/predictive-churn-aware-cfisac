module control_unit (
    input [3:0] opcode,
    input zero,
    output reg reg_write,
    output reg alu_src,
    output reg mem_read,
    output reg mem_write,
    output reg mem_to_reg,
    output reg [3:0] alu_control,
    output reg pc_src,
    output reg halt
);
    always @(*) begin
        // Defaults
        reg_write = 0; alu_src = 0; mem_read = 0; mem_write = 0; 
        mem_to_reg = 0; pc_src = 0; halt = 0; alu_control = opcode;

        case (opcode)
            4'b0000: ; // NOP
            4'b0001: begin reg_write = 1; alu_src = 0; end // MOV
            4'b0010: begin reg_write = 1; alu_src = 1; alu_control = 4'b0001; end // MVI (Pass Imm)
            4'b0011, 4'b0100, 4'b0101, 4'b0110, 4'b0111: begin // ADD, SUB, AND, OR, XOR
                reg_write = 1; alu_src = 0;
            end
            4'b1000: begin // LOAD
                reg_write = 1; alu_src = 1; mem_read = 1; mem_to_reg = 1; alu_control = 4'b0011; // ADD base+offset
            end
            4'b1001: begin // STORE
                alu_src = 1; mem_write = 1; alu_control = 4'b0011; // ADD base+offset
            end
            4'b1010: pc_src = 1; // JMP
            4'b1011: if (zero) pc_src = 1; // JZ
            4'b1100: if (!zero) pc_src = 1; // JNZ
            4'b1101, 4'b1110: begin // SHL, SHR
                reg_write = 1; alu_src = 1;
            end
            4'b1111: halt = 1; // HALT
            default: ;
        endcase
    end
endmodule
