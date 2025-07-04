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

    static padShape(tA: Tensor, tB: Tensor): [Tensor[], Tensor[]] {
        let dimA = TensorMath.getShape(tA).length;
        let dimB = TensorMath.getShape(tB).length;

        while (dimA < dimB) {
            dimA++;
            tA = [tA];
        }

        while (dimA > dimB) {
            dimB++;
            tB = [tB];
        }

        return [tA as Tensor[], tB as Tensor[]];
    }

    static add(tA: Tensor, tB: Tensor): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA + tB;
        }

        [tA, tB] = TensorMath.padShape(tA, tB);
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
    }

    static sub(tA: Tensor, tB: Tensor): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA - tB;
        }

        [tA, tB] = TensorMath.padShape(tA, tB);
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
    }

    static mul(tA: Tensor, tB: Tensor): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA * tB;
        }

        [tA, tB] = TensorMath.padShape(tA, tB);
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
    }

    static pow(tA: Tensor, tB: Tensor): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA ** tB;
        }

        [tA, tB] = TensorMath.padShape(tA, tB);
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
    }

    static div(tA: Tensor, tB: Tensor): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA / tB;
        }

        [tA, tB] = TensorMath.padShape(tA, tB);
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
    }

    static gt(tA: Tensor, tB: Tensor): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA > tB ? 1 : 0;
        }

        [tA, tB] = TensorMath.padShape(tA, tB);
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
    }

    static lt(tA: Tensor, tB: Tensor): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA < tB ? 1 : 0;
        }

        [tA, tB] = TensorMath.padShape(tA, tB);
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
    }

    static ge(tA: Tensor, tB: Tensor): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA >= tB ? 1 : 0;
        }

        [tA, tB] = TensorMath.padShape(tA, tB);
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
    }

    static le(tA: Tensor, tB: Tensor): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA <= tB ? 1 : 0;
        }

        [tA, tB] = TensorMath.padShape(tA, tB);
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
    }

    static eq(tA: Tensor, tB: Tensor): Tensor {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA === tB ? 1 : 0;
        }

        [tA, tB] = TensorMath.padShape(tA, tB);
        const outLen = Math.max(tA.length, tB.length);

        if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
            throw new Error("Inputs are incompatible tensors");
        }

        const result: Tensor[] = [];
        
        for (let i = 0; i < outLen; i++) {
            const subA = tA[tA.length === 1 ? 0 : i];
            const subB = tB[tB.length === 1 ? 0 : i];
            result.push(TensorMath.eq(subA, subB));
        }

        return result;
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
