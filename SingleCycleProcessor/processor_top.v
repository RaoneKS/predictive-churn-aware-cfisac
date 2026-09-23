module processor_top (
    input clk,
    input reset
);
    // Wires
    wire [15:0] pc, next_pc, pc_plus_1, branch_target;
    wire [15:0] instruction;
    wire [15:0] read_data1, read_data2, write_data, alu_result, mem_read_data;
    wire [15:0] alu_operand2;
    wire [15:0] imm_ext;
    
    // Control Wires
    wire reg_write, alu_src, mem_read, mem_write, mem_to_reg, pc_src, halt, zero;
    wire [3:0] alu_control;
    
    // Instruction Decode extraction based on ISA table[cite: 1]
    wire [3:0] opcode = instruction[15:12];
    wire [2:0] rd_rs = instruction[11:9]; // Destination or Source for STORE
    wire [2:0] rs1_base = instruction[8:6];
    wire [2:0] rs2 = instruction[2:0];
    wire [4:0] imm = instruction[4:0];

    // Sign/Zero Extension for Immediate value (5 bits -> 16 bits)
    assign imm_ext = { {11{imm[4]}}, imm }; 

    // Datapath Routing (Multiplexers and Adders)[cite: 2]
    assign pc_plus_1 = pc + 1; // Word addressable mapping
    assign branch_target = pc_plus_1 + imm_ext; // Relative branching assumption
    assign next_pc = pc_src ? branch_target : pc_plus_1; 
    
    assign alu_operand2 = alu_src ? imm_ext : read_data2;
    assign write_data = mem_to_reg ? mem_read_data : alu_result;

    // Submodule Instantiations
    pc_reg PC (
        .clk(clk), .reset(reset), .halt(halt), .next_pc(next_pc), .pc(pc)
    );

    inst_mem IMEM (
        .read_addr(pc), .instruction(instruction)
    );

    control_unit CU (
        .opcode(opcode), .zero(zero), .reg_write(reg_write), .alu_src(alu_src),
        .mem_read(mem_read), .mem_write(mem_write), .mem_to_reg(mem_to_reg),
        .alu_control(alu_control), .pc_src(pc_src), .halt(halt)
    );

    reg_file REG (
        .clk(clk), .reg_write(reg_write),
        .read_reg1(rs1_base), 
        .read_reg2(opcode == 4'b1001 ? rd_rs : rs2), // STORE needs Rs, else Rs2[cite: 1]
        .write_reg(rd_rs), 
        .write_data(write_data),
        .read_data1(read_data1), .read_data2(read_data2)
    );

    alu ALU (
        .a(read_data1), .b(alu_operand2), .alu_control(alu_control),
        .result(alu_result), .zero(zero)
    );

    data_mem DMEM (
        .clk(clk), .mem_read(mem_read), .mem_write(mem_write),
        .address(alu_result), .write_data(read_data2), .read_data(mem_read_data)
    );

endmodule
