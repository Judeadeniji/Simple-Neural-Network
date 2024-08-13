import { array, maths } from "utiliti-js";
import { Activation, Loss } from "./activation-functions";
import { random } from "./data";
import { Matrix } from "./matrix";

const np = {
  randn(x: number, y: number) {
    return Array.from<number[]>({ length: x })
      .fill([])
      .map(() =>
        Array.from<number>({ length: y })
          .fill(0)
          .map(() => Math.random())
      );
  },
  zeros(shape: number | number[]) {
    if (typeof shape === "number") {
      return new Array<number>(shape).fill(0);
    } else if (shape instanceof Array) {
      return [...shape].reduceRight((acc, curr) => {
        return new Array(curr).fill(0).map(() => acc);
      }, random() as any);
    }
  },
};

function normalizer(data: number[][]) {
  const normalizedData: number[][] = [];

  for (let i = 0; i < data.length; i++) {
    const l_data = data[i];
    normalizedData[i] = [];
    const total = l_data.reduce((a, b) => a + b, 0);
    for (let j = 0; j < data[i].length; j++) {
      normalizedData[i][j] = data[i][j] / total;
    }
  }

  return normalizedData;
}

class NeuralNetwork {
  layers: number[];
  weights: number[][][] = [];
  biases: number[][][] = [];


  constructor(layers: number[], private options?: { activationFunction: (n: number) => number; activationFunctionDerivative: (n: number) => number }) {
    this.layers = layers;
    this.initializeWeights();
  }

  serialize() {
    return JSON.stringify({
      layers: this.layers,
      weights: this.weights,
      biases: this.biases,
    });
  }

  load(data: string) {
    const parsedData = JSON.parse(data);
    this.layers = parsedData.layers;
    this.weights = parsedData.weights;
    this.biases = parsedData.biases;
  }

  private initializeWeights() {
    for (let i = 0; i < this.layers.length - 1; i++) {
      const layerWeights: number[][] = Array.from(
        { length: this.layers[i + 1] },
        () =>
          Array.from({ length: this.layers[i] }, () => maths.random() * 2 - 1)
      );
      const layerBiases: number[][] = Array.from(
        { length: this.layers[i + 1] },
        () => [maths.random() * 2 - 1]
      );

      this.weights.push(layerWeights);
      this.biases.push(layerBiases);
    }
  }

  private forward(input: number[][]): {
    activations: number[][][];
    zs: number[][][];
  } {
    let activation = input;
    const activations = [input];
    const zs = [];

    for (let i = 0; i < this.weights.length; i++) {
      const z = Matrix.add(
        Matrix.dot(this.weights[i], activation),
        this.biases[i]
      );
      zs.push(z);
      activation = Matrix.applyFunction(z, this.options?.activationFunction || Activation.relu);
      activations.push(activation);
    }

    return { activations, zs };
  }

  private backward(
    activations: number[][][],
    zs: number[][][],
    y: number[][]
  ): { weightGradients: number[][][]; biasGradients: number[][][] } {
    const weightGradients = [];
    const biasGradients = [];

    let delta = Matrix.multiplyElementWise(
      Loss.mseDerivative(activations[activations.length - 1], y),
      Matrix.applyFunction(zs[zs.length - 1], this.options?.activationFunctionDerivative || Activation.reluDerivative)
    );
    weightGradients.push(
      Matrix.dot(delta, Matrix.transpose(activations[activations.length - 2]))
    );
    biasGradients.push(delta);

    for (let l = 2; l < this.layers.length; l++) {
      const z = zs[zs.length - l];
      const sp = Matrix.applyFunction(z, this.options?.activationFunctionDerivative || Activation.reluDerivative);
      delta = Matrix.multiplyElementWise(
        Matrix.dot(
          Matrix.transpose(this.weights[this.weights.length - l + 1]),
          delta
        ),
        sp
      );

      let acts: number[][] | number[][][] =
        activations[activations.length - l - 1];
      // if acts is a 1D array, convert it to a 2D array
      if (acts[0].length === undefined) {
        acts = acts.map((act) => [act]);
      }

      weightGradients.push(
        Matrix.dot(delta, Matrix.transpose(acts as number[][]))
      );
      biasGradients.push(delta);
    }

    weightGradients.reverse();
    biasGradients.reverse();

    return { weightGradients, biasGradients };
  }

  async train(
    inputs: number[][][],
    outputs: number[][][],
    epochs: number,
    learningRate: number,
    batchSize: number = epochs
  ) {
    const numBatches = Math.ceil(inputs.length / batchSize);

    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;

      // Process each batch in parallel
      const processBatch = async (batchIndex: number) => {
        const start = batchIndex * batchSize;
        const end = Math.min(start + batchSize, inputs.length);

        let batchLoss = 0;

        for (let i = start; i < end; i++) {
          const { activations, zs } = this.forward(inputs[i]);
          batchLoss += Loss.mse(
            activations[activations.length - 1],
            outputs[i]
          );

          const { weightGradients, biasGradients } = this.backward(
            activations,
            zs,
            outputs[i]
          );

          // Update weights and biases
          for (let j = 0; j < this.weights.length; j++) {
            this.weights[j] = Matrix.subtract(
              this.weights[j],
              Matrix.scalarMultiply(weightGradients[j], learningRate)
            );
            this.biases[j] = Matrix.subtract(
              this.biases[j],
              Matrix.scalarMultiply(biasGradients[j], learningRate)
            );
          }
        }

        return batchLoss;
      };

      // Execute all batches in parallel
      const batchLosses = await Promise.all(
        Array.from({ length: numBatches }, (_, i) => processBatch(i))
      );

      // Sum the losses
      totalLoss = batchLosses.reduce((acc, loss) => acc + loss, 0);

      console.log(`Epoch ${epoch + 1}, Loss: ${totalLoss / inputs.length}`);
    }
  }

  private shuffleData(inputs: number[][][], outputs: number[][][]) {
    // inputs are the same length as outputs, we shuffle keeping the index integrity
    function zip<T, U>(a: T[], b: U[]): [T, U][] {
      return a.map((e, i) => [e, b[i]]);
    }

    const zipped = zip(inputs, outputs);
    zipped.sort(() => maths.random() - 0.5);

    return {
      inputs: zipped.map((pair) => pair[0]),
      outputs: zipped.map((pair) => pair[1]),
    }
  }

  trainsync(
    { inputs, targets }:{inputs: number[][][],
    targets: number[][][],},
    epochs: number,
    learningRate: number,
  ) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
      const sD = this.shuffleData(inputs, targets);
      inputs = sD.inputs;
      targets = sD.outputs;

      for (let i = 0; i < inputs.length; i++) {
        const { activations, zs } = this.forward(inputs[i]);
        totalLoss += Loss.mse(activations[activations.length - 1], targets[i]);

        const { weightGradients, biasGradients } = this.backward(
          activations,
          zs,
          targets[i]
        );

        for (let j = 0; j < this.weights.length; j++) {
          this.weights[j] = Matrix.subtract(
            this.weights[j],
            Matrix.scalarMultiply(weightGradients[j], learningRate)
          );
          this.biases[j] = Matrix.subtract(
            this.biases[j],
            Matrix.scalarMultiply(biasGradients[j], learningRate)
          );
        }
      }

      console.log(`Epoch ${epoch + 1}, Loss: ${totalLoss / inputs.length}`);
    }
  }

  predict(input: number[][]): number[][] {
    return this.forward(input).activations[this.layers.length - 1];
  }
}

export { NeuralNetwork };
