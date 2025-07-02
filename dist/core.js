"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Node = exports.OP = void 0;
var OP;
(function (OP) {
    OP[OP["NONE"] = 0] = "NONE";
    OP[OP["ADD"] = 1] = "ADD";
    OP[OP["MUL"] = 2] = "MUL";
    OP[OP["POW"] = 3] = "POW";
    OP[OP["DIV"] = 4] = "DIV";
    OP[OP["NEG"] = 5] = "NEG";
    OP[OP["EXP"] = 6] = "EXP";
    OP[OP["LOG"] = 7] = "LOG";
    OP[OP["RELU"] = 8] = "RELU";
    OP[OP["SIGMOID"] = 9] = "SIGMOID";
    OP[OP["TANH"] = 10] = "TANH";
})(OP || (exports.OP = OP = {}));
class Node {
    // Only scalars are supported for now
    value;
    grad;
    children;
    feedBackward;
    op;
    constructor(value, children = [], op = OP.NONE) {
        this.value = value;
        this.grad = 0;
        this.children = children;
        this.op = op;
        this.feedBackward = () => { };
    }
    add(other) {
        other = this.forceNode(other);
        const out = new Node(this.value + other.value, [this, other], OP.ADD);
        out.feedBackward = () => {
            this.grad += out.grad;
            other.grad += out.grad;
        };
        return out;
    }
    mul(other) {
        other = this.forceNode(other);
        const out = new Node(this.value * other.value, [this, other], OP.MUL);
        out.feedBackward = () => {
            this.grad += out.grad * other.value;
            other.grad += out.grad * this.value;
        };
        return out;
    }
    pow(other) {
        if (other instanceof Node) {
            const out = new Node(this.value ** other.value, [this, other], OP.POW);
            out.feedBackward = () => {
                this.grad += out.grad * other.value * this.value ** (other.value - 1);
                other.grad += out.grad * this.value ** other.value * Math.log(this.value);
            };
            return out;
        }
        const out = new Node(this.value ** other, [this], OP.POW);
        out.feedBackward = () => {
            this.grad += out.grad * other * this.value ** (other - 1);
        };
        return out;
    }
    div(other) {
        other = this.forceNode(other);
        const out = new Node(this.value / other.value, [this, other], OP.DIV);
        out.feedBackward = () => {
            this.grad += out.grad * (1 / other.value);
            other.grad += out.grad * -this.value / other.value ** 2;
        };
        return out;
    }
    neg() {
        const out = new Node(-this.value, [this], OP.NEG);
        out.feedBackward = () => {
            this.grad += -out.grad;
        };
        return out;
    }
    exp() {
        const exp = Math.exp(this.value);
        const out = new Node(exp, [this], OP.EXP);
        out.feedBackward = () => {
            this.grad += exp;
        };
        return out;
    }
    log() {
        const out = new Node(Math.log(this.value), [this], OP.LOG);
        out.feedBackward = () => {
            this.grad += 1 / this.value;
        };
        return out;
    }
    relu() {
        const out = new Node(Math.max(this.value, 0), [this], OP.RELU);
        out.feedBackward = () => {
            this.grad += out.grad * (this.value < 0 ? 0 : 1);
        };
        return out;
    }
    sigmoid() {
        const sigmoid = 1 / (1 + Math.exp(-this.value));
        const out = new Node(sigmoid, [this], OP.SIGMOID);
        out.feedBackward = () => {
            this.grad += out.grad * sigmoid * (1 - sigmoid);
        };
        return out;
    }
    tanh() {
        const tanh = Math.tanh(this.value);
        const out = new Node(tanh, [this], OP.TANH);
        out.feedBackward = () => {
            this.grad += out.grad * (1 - tanh * tanh);
        };
        return out;
    }
    backward() {
        // Build topological order
        const topo = [];
        const visited = new Set();
        function build(node) {
            if (!visited.has(node)) {
                visited.add(node);
                node.grad = 0;
                for (let child of node.children)
                    build(child);
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
    forceNode(value) {
        if (value instanceof Node)
            return value;
        return new Node(value);
    }
}
exports.Node = Node;
