/* Quartus Prime Chain Description File */
JedecChain;
    FileRevision(JESD32A);
    DefaultMfr(6E);

    P ActionCode(Ign)
        Device PartName(SOCVHPS) MfrSpec(OpMask(0));

    P ActionCode(Cfg)
        Device PartName(5CSXFC6D6F31C6)
        Path("/home/raone/fpga-ai-accelerator/quartus/de10_standard_hw/output_files/")
        File("de10_standard_hw.sof")
        MfrSpec(OpMask(1));

ChainEnd;

AlteraBegin;
    ChainType(JTAG);
AlteraEnd;
