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
  
  constructor(layers: number[]) {
    this.layers = layers;
    this.initializeWeights();
  }
  private initializeWeights() {
      for (let i = 0; i < this.layers.length - 1; i++) {
          const layerWeights: number[][] = Array.from({ length: this.layers[i + 1] }, () =>
              Array.from({ length: this.layers[i] }, () => Math.random() * 2 - 1)
          );
          const layerBiases: number[][] = Array.from({ length: this.layers[i + 1] }, () => [Math.random() * 2 - 1]);
  
          this.weights.push(layerWeights);
          this.biases.push(layerBiases);
      }
  }

  private forward(input: number[][]): { activations: number[][][]; zs: number[][][] } {
      let activation = input;
      const activations = [input];
      const zs = [];

      for (let i = 0; i < this.weights.length; i++) {
          const z = Matrix.add(Matrix.dot(this.weights[i], activation), this.biases[i]);
          zs.push(z);
          activation = Matrix.applyFunction(z, Activation.sigmoid);
          activations.push(activation);
      }

      return { activations, zs };
  }

  private backward(activations: number[][][], zs: number[][][], y: number[][]): { weightGradients: number[][][]; biasGradients: number[][][] } {
      const weightGradients = [];
      const biasGradients = [];

      let delta = Matrix.multiplyElementWise(Loss.mseDerivative(activations[activations.length - 1], y), Matrix.applyFunction(zs[zs.length - 1], Activation.sigmoidDerivative));
      weightGradients.push(Matrix.dot(delta, Matrix.transpose(activations[activations.length - 2])));
      biasGradients.push(delta);

      for (let l = 2; l < this.layers.length; l++) {
          const z = zs[zs.length - l];
          const sp = Matrix.applyFunction(z, Activation.sigmoidDerivative);
          delta = Matrix.multiplyElementWise(Matrix.dot(Matrix.transpose(this.weights[this.weights.length - l + 1]), delta), sp);
          weightGradients.push(Matrix.dot(delta, Matrix.transpose(activations[activations.length - l - 1])));
          biasGradients.push(delta);
      }

      weightGradients.reverse();
      biasGradients.reverse();

      return { weightGradients, biasGradients };
  }

  train(inputs: number[][][], outputs: number[][][], epochs: number, learningRate: number) {
      for (let epoch = 0; epoch < epochs; epoch++) {
          let totalLoss = 0;

          for (let i = 0; i < inputs.length; i++) {
              const { activations, zs } = this.forward(inputs[i]);
              totalLoss += Loss.mse(activations[activations.length - 1], outputs[i]);

              const { weightGradients, biasGradients } = this.backward(activations, zs, outputs[i]);

              for (let j = 0; j < this.weights.length; j++) {
                  this.weights[j] = Matrix.subtract(this.weights[j], Matrix.scalarMultiply(weightGradients[j], learningRate));
                  this.biases[j] = Matrix.subtract(this.biases[j], Matrix.scalarMultiply(biasGradients[j], learningRate));
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
