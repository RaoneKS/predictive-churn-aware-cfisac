# The accelerator clock port is constrained at 100 MHz.
create_clock -name clk -period 10.000 [get_ports {clk}]
