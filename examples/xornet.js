const { Tensor } = require("../index"), rand = () => Math.random() * 2 - 1;

class Xornet {
    constructor(options = {}) {
        // 2->2->1 xornet
        this.w1 = new Tensor(options.w1 || [
            [rand(), rand()],
            [rand(), rand()]
        ], { requiresGrad: true });
        this.b1 = new Tensor(options.b1 || [0, 0], { requiresGrad: true });
        this.w2 = new Tensor(options.w2 || [
            [rand()],
            [rand()]
        ], { requiresGrad: true });
        this.b2 = new Tensor(options.b2 || [0], { requiresGrad: true });
        this.lr = options.lr || 0.5;
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

        // We disable gradient collecting first to calculate new weight, then enable it for next pass
        this.w1 = this.w1.withGrad(false).sub(this.w1.grad.mul(this.lr)).withGrad(true);
        this.w2 = this.w2.withGrad(false).sub(this.w2.grad.mul(this.lr)).withGrad(true);
        this.b1 = this.b1.withGrad(false).sub(this.b1.grad.mul(this.lr)).withGrad(true);
        this.b2 = this.b2.withGrad(false).sub(this.b2.grad.mul(this.lr)).withGrad(true);
    }
}

const xornet = new Xornet();

for (let epoch = 0; epoch < 30000; epoch++) {
    xornet.backprop([1,0], [1]);
    xornet.backprop([0,1], [1]);
    xornet.backprop([0,0], [0]);
    xornet.backprop([1,1], [0]);
}

console.log(xornet.forward([1,1]).val()); // 0-ish
console.log(xornet.forward([1,0]).val()); // 1-ish
console.log(xornet.forward([0,1]).val()); // 1-ish
console.log(xornet.forward([0,0]).val()); // 0-ish
