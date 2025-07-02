export enum OP {
    NONE,
    ADD,
    MUL,
    POW,
    DIV,
    NEG,
    EXP,
    LOG,
    RELU,
    SIGMOID,
    TANH
}

export class Node {
    // Only scalars are supported for now
    public value: number;
    public grad: number;
    public children: Node[];
    public feedBackward: Function;
    public op: OP;

    constructor(value: number, children: Node[] = [], op: OP = OP.NONE) {
        this.value = value;
        this.grad = 0;
        this.children = children;
        this.op = op;
        this.feedBackward = () => {};
    }

    add(other: Node | number): Node {
        other = this.forceNode(other);
        const out = new Node(this.value + other.value, [this, other], OP.ADD);

        out.feedBackward = () => {
            this.grad += out.grad;
            other.grad += out.grad;
        }

        return out;
    }

    mul(other: Node | number): Node {
        other = this.forceNode(other);
        const out = new Node(this.value * other.value, [this, other], OP.MUL);

        out.feedBackward = () => {
            this.grad += out.grad * other.value;
            other.grad += out.grad * this.value;
        }

        return out;
    }

    pow(other: Node | number): Node {
        if (other instanceof Node) {
            const out = new Node(this.value ** other.value, [this, other], OP.POW);

            out.feedBackward = () => {
                this.grad += out.grad * other.value * this.value ** (other.value - 1);
                other.grad += out.grad * this.value ** other.value * Math.log(this.value);
            }

            return out;
        }

        const out = new Node(this.value ** other, [this], OP.POW);

        out.feedBackward = () => {
            this.grad += out.grad * other * this.value ** (other - 1);
        }

        return out;
    }

    div(other: Node | number): Node {
        other = this.forceNode(other);
        const out = new Node(this.value / other.value, [this, other], OP.DIV);

        out.feedBackward = () => {
            this.grad += out.grad / other.value;
            other.grad += out.grad * -this.value / other.value**2;
        }

        return out;
    }

    neg(): Node {
        const out = new Node(-this.value, [this], OP.NEG);

        out.feedBackward = () => {
            this.grad += -out.grad;
        }

        return out;
    }

    exp(): Node {
        const exp = Math.exp(this.value);
        const out = new Node(exp, [this], OP.EXP);

        out.feedBackward = () => {
            this.grad += out.grad * exp;
        }

        return out;
    }

    log(): Node {
        const out = new Node(Math.log(this.value), [this], OP.LOG);

        out.feedBackward = () => {
            this.grad += out.grad / this.value;
        }

        return out;
    }

    relu(): Node {
        const out = new Node(Math.max(this.value, 0), [this], OP.RELU);
        out.feedBackward = () => {
            this.grad += out.grad * (this.value < 0 ? 0 : 1);
        };

        return out;
    }

    sigmoid(): Node {
        const sigmoid = 1 / (1 + Math.exp(-this.value));

        const out = new Node(sigmoid, [this], OP.SIGMOID);
        out.feedBackward = () => {
            this.grad += out.grad * sigmoid * (1 - sigmoid);
        };

        return out;
    }

    tanh(): Node {
        const tanh = Math.tanh(this.value);

        const out = new Node(tanh, [this], OP.TANH);
        out.feedBackward = () => {
            this.grad += out.grad * (1 - tanh * tanh);
        };

        return out;
    }

    backward() {
        // Build topological order
        const topo: Node[] = [];
        const visited: Set<Node> = new Set();

        function build(node: Node) {
            if (!visited.has(node)) {
                visited.add(node);
                node.grad = 0;

                for (let child of node.children) build(child);
                topo.push(node);
            }
        }

        build(this);

        // Feed backward to calculate gradient
        this.grad = 1; // Derivative of itself with respect to itself

        for (let index = topo.length - 1; index > -1; index--) {
            topo[index].feedBackward();
        }
    }

    forceNode(value: Node | number): Node {
        if (value instanceof Node) return value;

        return new Node(value);
    }
}
