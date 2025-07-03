export type Tensor = number | Tensor[];

export class TensorMath {
    static create(num: number, shape: number[]): Tensor {
        if (shape.length === 0) {
            return num;
        }

        const [dim, ...rest] = shape;
        const out = [];

        for (let i = 0; i < dim; i++) {
            out.push(TensorMath.create(num, rest));
        }

        return out;
    }
    
    static getShape(tA: Tensor): number[] {
        const shape: number[] = [];
        
        let subA = tA;

        while (Array.isArray(subA)) {
            shape.push(subA.length);
            subA = subA[0];
        }

        return shape;
    }

    static add(tA: Tensor, tB: Tensor): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA + tB;
        } else if (Array.isArray(tA) && Array.isArray(tB)) {
            const outLen = Math.max(tA.length, tB.length);

            if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
                throw new Error("Inputs are incompatible tensors");
            }

            const result: Tensor[] = [];
            
            for (let i = 0; i < outLen; i++) {
                const subA = tA[tA.length === 1 ? 0 : i];
                const subB = tB[tB.length === 1 ? 0 : i];
                result.push(TensorMath.add(subA, subB));
            }

            return result;
        } else if (Array.isArray(tA) && typeof tB === "number") {
            return tA.map(subA => TensorMath.add(subA, tB));
        } else if (typeof tA === "number" && Array.isArray(tB)) {
            return tB.map(subB => TensorMath.add(tA, subB));
        }

        throw new Error("Inputs are not tensors");
    }

    static sub(tA: Tensor, tB: Tensor): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA - tB;
        } else if (Array.isArray(tA) && Array.isArray(tB)) {
            const outLen = Math.max(tA.length, tB.length);

            if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
                throw new Error("Inputs are incompatible tensors");
            }

            const result: Tensor[] = [];
            
            for (let i = 0; i < outLen; i++) {
                const subA = tA[tA.length === 1 ? 0 : i];
                const subB = tB[tB.length === 1 ? 0 : i];
                result.push(TensorMath.sub(subA, subB));
            }

            return result;
        } else if (Array.isArray(tA) && typeof tB === "number") {
            return tA.map(subA => TensorMath.sub(subA, tB));
        } else if (typeof tA === "number" && Array.isArray(tB)) {
            return tB.map(subB => TensorMath.sub(tA, subB));
        }

        throw new Error("Inputs are not tensors");
    }

    static mul(tA: Tensor, tB: Tensor): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA * tB;
        } else if (Array.isArray(tA) && Array.isArray(tB)) {
            const outLen = Math.max(tA.length, tB.length);

            if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
                throw new Error("Inputs are incompatible tensors");
            }

            const result: Tensor[] = [];
            
            for (let i = 0; i < outLen; i++) {
                const subA = tA[tA.length === 1 ? 0 : i];
                const subB = tB[tB.length === 1 ? 0 : i];
                result.push(TensorMath.mul(subA, subB));
            }

            return result;
        } else if (Array.isArray(tA) && typeof tB === "number") {
            return tA.map(subA => TensorMath.mul(subA, tB));
        } else if (typeof tA === "number" && Array.isArray(tB)) {
            return tB.map(subB => TensorMath.mul(tA, subB));
        }

        throw new Error("Inputs are not tensors");
    }

    static pow(tA: Tensor, tB: Tensor): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA ** tB;
        } else if (Array.isArray(tA) && Array.isArray(tB)) {
            const outLen = Math.max(tA.length, tB.length);

            if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
                throw new Error("Inputs are incompatible tensors");
            }

            const result: Tensor[] = [];
            
            for (let i = 0; i < outLen; i++) {
                const subA = tA[tA.length === 1 ? 0 : i];
                const subB = tB[tB.length === 1 ? 0 : i];
                result.push(TensorMath.pow(subA, subB));
            }

            return result;
        } else if (Array.isArray(tA) && typeof tB === "number") {
            return tA.map(subA => TensorMath.pow(subA, tB));
        } else if (typeof tA === "number" && Array.isArray(tB)) {
            return tB.map(subB => TensorMath.pow(tA, subB));
        }

        throw new Error("Inputs are not tensors");
    }

    static div(tA: Tensor, tB: Tensor): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA / tB;
        } else if (Array.isArray(tA) && Array.isArray(tB)) {
            const outLen = Math.max(tA.length, tB.length);

            if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
                throw new Error("Inputs are incompatible tensors");
            }

            const result: Tensor[] = [];
            
            for (let i = 0; i < outLen; i++) {
                const subA = tA[tA.length === 1 ? 0 : i];
                const subB = tB[tB.length === 1 ? 0 : i];
                result.push(TensorMath.div(subA, subB));
            }

            return result;
        } else if (Array.isArray(tA) && typeof tB === "number") {
            return tA.map(subA => TensorMath.div(subA, tB));
        } else if (typeof tA === "number" && Array.isArray(tB)) {
            return tB.map(subB => TensorMath.div(tA, subB));
        }

        throw new Error("Inputs are not tensors");
    }

    static gt(tA: Tensor, tB: Tensor): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA > tB ? 1 : 0;
        } else if (Array.isArray(tA) && Array.isArray(tB)) {
            const outLen = Math.max(tA.length, tB.length);

            if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
                throw new Error("Inputs are incompatible tensors");
            }

            const result: Tensor[] = [];
            
            for (let i = 0; i < outLen; i++) {
                const subA = tA[tA.length === 1 ? 0 : i];
                const subB = tB[tB.length === 1 ? 0 : i];
                result.push(TensorMath.gt(subA, subB));
            }

            return result;
        } else if (Array.isArray(tA) && typeof tB === "number") {
            return tA.map(subA => TensorMath.gt(subA, tB));
        } else if (typeof tA === "number" && Array.isArray(tB)) {
            return tB.map(subB => TensorMath.gt(tA, subB));
        }

        throw new Error("Inputs are not tensors");
    }

    static lt(tA: Tensor, tB: Tensor): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA < tB ? 1 : 0;
        } else if (Array.isArray(tA) && Array.isArray(tB)) {
            const outLen = Math.max(tA.length, tB.length);

            if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
                throw new Error("Inputs are incompatible tensors");
            }

            const result: Tensor[] = [];
            
            for (let i = 0; i < outLen; i++) {
                const subA = tA[tA.length === 1 ? 0 : i];
                const subB = tB[tB.length === 1 ? 0 : i];
                result.push(TensorMath.lt(subA, subB));
            }

            return result;
        } else if (Array.isArray(tA) && typeof tB === "number") {
            return tA.map(subA => TensorMath.lt(subA, tB));
        } else if (typeof tA === "number" && Array.isArray(tB)) {
            return tB.map(subB => TensorMath.lt(tA, subB));
        }

        throw new Error("Inputs are not tensors");
    }

    static ge(tA: Tensor, tB: Tensor): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA >= tB ? 1 : 0;
        } else if (Array.isArray(tA) && Array.isArray(tB)) {
            const outLen = Math.max(tA.length, tB.length);

            if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
                throw new Error("Inputs are incompatible tensors");
            }

            const result: Tensor[] = [];
            
            for (let i = 0; i < outLen; i++) {
                const subA = tA[tA.length === 1 ? 0 : i];
                const subB = tB[tB.length === 1 ? 0 : i];
                result.push(TensorMath.ge(subA, subB));
            }

            return result;
        } else if (Array.isArray(tA) && typeof tB === "number") {
            return tA.map(subA => TensorMath.ge(subA, tB));
        } else if (typeof tA === "number" && Array.isArray(tB)) {
            return tB.map(subB => TensorMath.ge(tA, subB));
        }

        throw new Error("Inputs are not tensors");
    }

    static le(tA: Tensor, tB: Tensor): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA <= tB ? 1 : 0;
        } else if (Array.isArray(tA) && Array.isArray(tB)) {
            const outLen = Math.max(tA.length, tB.length);

            if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
                throw new Error("Inputs are incompatible tensors");
            }

            const result: Tensor[] = [];
            
            for (let i = 0; i < outLen; i++) {
                const subA = tA[tA.length === 1 ? 0 : i];
                const subB = tB[tB.length === 1 ? 0 : i];
                result.push(TensorMath.le(subA, subB));
            }

            return result;
        } else if (Array.isArray(tA) && typeof tB === "number") {
            return tA.map(subA => TensorMath.le(subA, tB));
        } else if (typeof tA === "number" && Array.isArray(tB)) {
            return tB.map(subB => TensorMath.le(tA, subB));
        }

        throw new Error("Inputs are not tensors");
    }

    static neg(tA: Tensor): Tensor {
        if (typeof tA === "number") {
            return -tA;
        } else {
            return tA.map(subA => TensorMath.neg(subA));
        }
    }

    static exp(tA: Tensor): Tensor {
        if (typeof tA === "number") {
            return Math.exp(tA);
        } else {
            return tA.map(subA => TensorMath.exp(subA));
        }
    }

    static log(tA: Tensor): Tensor {
        if (typeof tA === "number") {
            return Math.log(tA);
        } else {
            return tA.map(subA => TensorMath.log(subA));
        }
    }

    static relu(tA: Tensor): Tensor {
        if (typeof tA === "number") {
            return Math.max(tA, 0);
        } else {
            return tA.map(subA => TensorMath.relu(subA));
        }
    }

    static sigmoid(tA: Tensor): Tensor {
        if (typeof tA === "number") {
            return 1 / (1 + Math.exp(-tA));
        } else {
            return tA.map(subA => TensorMath.sigmoid(subA));
        }
    }

    static tanh(tA: Tensor): Tensor {
        if (typeof tA === "number") {
            return Math.tanh(tA);
        } else {
            return tA.map(subA => TensorMath.tanh(subA));
        }
    }
}
