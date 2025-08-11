const { Tensor, Optim, nn } = require("../index");

class Xornet {
    constructor(options = {}) {
        // 2->2->1 xornet
        this.l1 = new nn.Linear(2, 2);
        this.l2 = new nn.Linear(2, 1);
        // We use a simple SGD optimizer for this
        this.lr = options.lr || 0.5;
        this.optim = new Optim.SGD(nn.state.getParameters(this), { lr: this.lr });
    }

    forward(input) {
        return this.l2.forward(this.l1.forward(new Tensor(input)).sigmoid()).sigmoid();
    }

    backprop(input, target) {
        const T = new Tensor(target);
        const Y = this.forward(input);
        const L = Y.sub(T).pow(2);

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
