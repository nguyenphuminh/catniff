# mingpu

Mingpu is an extremely minimal GPU architecture that I wrote for fun and also to learn basic hardware design. Though, it does not have the "G" in "GPU" (yet) - it does not do any graphics, but rather focuses on pure parallel SIMD compute.

## Architecture

Currently, Mingpu includes:

* A control unit to dispatch instructions to compute cores.
* 80 compute cores each with:
    * 1 register/accumulator.
    * 16-bit word size in register and memory.
    * 8-bit opcode and 8-bit operand.
    * 512 bytes (256 16-bit words) of local mem each core.
    * A minimal 6-op ISA - NOP, ADD (signed), MUL (signed), LOAD, STORE, HALT.
* 1 8-bit program counter for all cores, which means 256 instructions (512 bytes) for kernel maximum.

## Setup

I currently use [Icarus Verilog](https://github.com/steveicarus/iverilog) for development of this project, so have it installed and you are good to go.

## Run testbench

```sh
make sim
```

## Configuration

You can configure the gpu (number of cores, local mem size, data width, etc.) in `./rtl/gpu_pkg.sv`.

## Todos

* Rethink better arch overall, currently this is a very naive arch and implementation from me. Though it should always stay minimal.
* Integration with real hardware, possibly an EBAZ4205, which should come with:
    * Ethernet communication code.
    * Driver.
    * Assembler.

## Copyright and License

Copyright © 2026 Nguyen Phu Minh.

This project is licensed under the Apache 2.0 License.
