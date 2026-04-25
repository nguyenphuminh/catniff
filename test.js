const { Tensor, Conv2d } = require("./dist/index");

// ── helpers ───────────────────────────────────────────────────────────────────

const seq   = (n, scale = 1) => Array.from({ length: n }, (_, i) => (i + 1) * scale);
const head  = t => { console.log('\n' + '═'.repeat(56)); console.log('  ' + t); console.log('═'.repeat(56)); };
const label = l => console.log('\n  · ' + l);

// ═════════════════════════════════════════════════════════
//  TEST 1  Basic forward + backward
// ═════════════════════════════════════════════════════════
head('TEST 1 · basic  [1,1,4,4] ✕ [1,1,3,3]  s=1 p=0 d=1 g=1');
{
    const input  = new Tensor(seq(16, 0.1), { shape: [1,1,4,4], requiresGrad: true });
    const weight = new Tensor(seq(9,  0.1), { shape: [1,1,3,3], requiresGrad: true });

    label('input');  console.log(input.toString());
    label('weight'); console.log(weight.toString());

    const out = input.conv2d(weight);
    label('output  (expect [[[[3.4800, 3.9300], [5.2800, 5.7300]]]])');
    console.log(out.toString());

    out.sum().backward();
    label('input.grad');  console.log(input.grad.toString());
    label('weight.grad'); console.log(weight.grad.toString());
}

// ═════════════════════════════════════════════════════════
//  TEST 2  Stride + padding + bias  — forward + backward
// ═════════════════════════════════════════════════════════
head('TEST 2 · stride+pad+bias  [2,3,5,5] ✕ [4,3,3,3]  s=2 p=1');
{
    const input  = new Tensor(seq(2*3*5*5, 0.02), { shape: [2,3,5,5], requiresGrad: true });
    const weight = new Tensor(seq(4*3*3*3, 0.01), { shape: [4,3,3,3], requiresGrad: true });
    const bias   = new Tensor([0.1, 0.2, 0.3, 0.4], { shape: [4],     requiresGrad: true });

    const out = input.conv2d(weight, bias, 2, 1);
    label('output shape should be [2,4,3,3]');
    console.log(out.toString());

    out.sum().backward();
    label('input.grad');  console.log(input.grad.toString());
    label('weight.grad'); console.log(weight.grad.toString());
    label('bias.grad  (each entry = N*Hout*Wout = 2*3*3 = 18)');
    console.log(bias.grad.toString());
}

// ═════════════════════════════════════════════════════════
//  TEST 3  Dilation  — forward + backward
// ═════════════════════════════════════════════════════════
head('TEST 3 · dilation  [1,2,7,7] ✕ [2,2,3,3]  d=2');
{
    const input  = new Tensor(seq(1*2*7*7, 0.01), { shape: [1,2,7,7], requiresGrad: true });
    const weight = new Tensor(seq(2*2*3*3, 0.05), { shape: [2,2,3,3], requiresGrad: true });

    const out = input.conv2d(weight, null, 1, 0, 2);
    label('output shape should be [1,2,3,3]');
    console.log(out.toString());

    out.sum().backward();
    label('input.grad');  console.log(input.grad.toString());
    label('weight.grad'); console.log(weight.grad.toString());
}

// ═════════════════════════════════════════════════════════
//  TEST 4  Depthwise (groups=Cin)  — forward + backward
// ═════════════════════════════════════════════════════════
head('TEST 4 · depthwise  [1,4,5,5] ✕ [4,1,3,3]  g=4');
{
    const input  = new Tensor(seq(1*4*5*5, 0.02), { shape: [1,4,5,5], requiresGrad: true });
    const weight = new Tensor(seq(4*1*3*3, 0.05), { shape: [4,1,3,3], requiresGrad: true });

    const out = input.conv2d(weight, null, 1, 0, 1, 4);
    label('output shape should be [1,4,3,3]');
    console.log(out.toString());

    out.sum().backward();
    label('input.grad');  console.log(input.grad.toString());
    label('weight.grad'); console.log(weight.grad.toString());
}

// ═════════════════════════════════════════════════════════
//  TEST 5a  Non-contiguous input — transpose(2, 3)
//
//  base: [1,1,4,6]  contiguous, strides [24, 24, 6, 1]
//  input = base.transpose(2,3) -> shape [1,1,6,4], strides [24, 24, 1, 6]
//  H-stride and W-stride are swapped: definitely non-contiguous
// ═════════════════════════════════════════════════════════
head('TEST 5a · non-contiguous input via transpose(2,3)');
{
    const base   = new Tensor(seq(1*1*4*6, 0.1), { shape: [1,1,4,6], requiresGrad: true });
    const input  = base.transpose(2, 3);   // [1,1,6,4], strides [24,24,1,6]
    const weight = new Tensor(seq(1*1*3*3, 0.1), { shape: [1,1,3,3], requiresGrad: true });

    label('base (contiguous [1,1,4,6])'); console.log(base.toString());
    label('input after transpose ([1,1,6,4], non-contiguous)'); console.log(input.toString());

    const out = input.conv2d(weight);
    label('output shape should be [1,1,4,2]');
    console.log(out.toString());

    out.sum().backward();
    label('base.grad');   console.log(base.grad.toString());
    label('weight.grad'); console.log(weight.grad.toString());
}

// ═════════════════════════════════════════════════════════
//  TEST 5b  Non-contiguous input — slice with step=2 on H and W
//
//  base: [1,1,8,8]  contiguous
//  input = base[:, :, ::2, ::2] -> shape [1,1,4,4]
//  physical H-stride = 16, W-stride = 2  (both doubled, non-contiguous)
// ═════════════════════════════════════════════════════════
head('TEST 5b · non-contiguous input via slice step=2 on H and W');
{
    const base   = new Tensor(seq(1*1*8*8, 0.05), { shape: [1,1,8,8], requiresGrad: true });
    const input  = base.slice([[0,1],[0,1],[0,8,2],[0,8,2]]);  // [1,1,4,4], non-contiguous
    const weight = new Tensor(seq(1*1*3*3, 0.1), { shape: [1,1,3,3], requiresGrad: true });

    label('base (contiguous [1,1,8,8])'); console.log(base.toString());
    label('input after slice ([1,1,4,4], non-contiguous)'); console.log(input.toString());

    const out = input.conv2d(weight);
    label('output shape should be [1,1,2,2]');
    console.log(out.toString());

    out.sum().backward();
    label('base.grad (non-zero only at even positions)'); console.log(base.grad.toString());
    label('weight.grad'); console.log(weight.grad.toString());
}

// ═════════════════════════════════════════════════════════
//  TEST 5c  Non-contiguous weight — permute from [kH,kW,Cout,Cin]
//
//  wbase stored as [3,3,2,2] in row-major
//  weight = wbase.permute([2,3,0,1]) -> [2,2,3,3], non-contiguous
//  strides go from [12,4,2,1]  to  [2,1,4,2]  (out-of-order)
// ═════════════════════════════════════════════════════════
head('TEST 5c · non-contiguous weight via permute([2,3,0,1])');
{
    const input  = new Tensor(seq(1*2*5*5, 0.05), { shape: [1,2,5,5], requiresGrad: true });
    const wbase  = new Tensor(seq(3*3*2*2, 0.1),  { shape: [3,3,2,2], requiresGrad: true });
    const weight = wbase.permute([2, 3, 0, 1]);   // [2,2,3,3], non-contiguous

    label('wbase (contiguous [3,3,2,2])'); console.log(wbase.toString());
    label('weight after permute ([2,2,3,3], non-contiguous)'); console.log(weight.toString());

    const out = input.conv2d(weight);
    label('output shape should be [1,2,3,3]');
    console.log(out.toString());

    out.sum().backward();
    label('input.grad');  console.log(input.grad.toString());
    label('wbase.grad');  console.log(wbase.grad.toString());
}

head('TEST 6 · nn.Conv2d  [2,3,5,5]  3→8  k=3 s=1 p=1');
{
    const layer = new Conv2d(3, 8, 3, 1, 1);
    const input = new Tensor(seq(2*3*5*5, 0.01), { shape: [2,3,5,5], requiresGrad: true });

    const out = layer.forward(input);
    label('output shape should be [2,8,5,5]');
    console.log(out.toString());

    out.sum().backward();
    label('input.grad shape should be [2,3,5,5]');
    console.log(input.grad.toString());
    label('weight.grad shape should be [8,3,3,3]');
    console.log(layer.weight.grad.toString());
    label('bias.grad shape should be [8]  (each = N*H*W = 2*5*5 = 50)');
    console.log(layer.bias.grad.toString());
}