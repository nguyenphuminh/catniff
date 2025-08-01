const { Tensor, Optim } = require("../index");

class Xornet {
    constructor(options = {}) {
        // 2->2->1 xornet
        this.w1 = Tensor.rand([2, 2], { requiresGrad: true });
        this.b1 = Tensor.zeros([2], { requiresGrad: true });
        this.w2 = Tensor.rand([2, 1], { requiresGrad: true });
        this.b2 = Tensor.zeros([1], { requiresGrad: true });
        this.lr = options.lr || 0.5;
        // We use simple SGD optimizer for this
        this.optim = new Optim.SGD([ this.w1, this.b1, this.w2, this.b2 ], { lr: this.lr });
    }

    forward(input) {
        return new Tensor(input)
                    .matmul(this.w1)
                    .add(this.b1)
                    .sigmoid()
                    .matmul(this.w2)
                    .add(this.b2)
                    .sigmoid();
    }

    backprop(input, target) {
        const T = new Tensor(target);
        const Y = this.forward(input);
        const L = Y.sub(T).pow(2).mul(0.5);

        L.backward();

        this.optim.step();
    }
}

const xornet = new Xornet();

const start = performance.now()

for (let epoch = 0; epoch < 30000; epoch++) {
    xornet.backprop([1,0], [1]);
    xornet.backprop([0,1], [1]);
    xornet.backprop([0,0], [0]);
    xornet.backprop([1,1], [0]);
}

console.log(`Finished training in ${performance.now() - start} ms`);

console.log(xornet.forward([1,1]).val()); // 0-ish
console.log(xornet.forward([1,0]).val()); // 1-ish
console.log(xornet.forward([0,1]).val()); // 1-ish
console.log(xornet.forward([0,0]).val()); // 0-ish
