import { array, maths } from "utiliti-js";
import { NeuralNetwork } from "./nn";
import { Activation, relu, sigmoid } from "./activation-functions";
import { wordVecs as wordVectors } from "./w2v";
import { dataPoints } from "./docs";
import readline from "node:readline"

interface TextDataPoint {
  text: string;
  class: -1 | 0 | 1; // -1 for Negative, 0 for Neutral, 1 for Positive
}

function generateSentimentDataset(): TextDataPoint[] {
  return array.shuffle([
    ...dataPoints,
    {
      text: "Although I initially had reservations, I now absolutely love this experience!",
      class: 1,
    },
    {
      text: "Despite the effort put into it, the result is utterly terrible and disappointing.",
      class: -1,
    },
    {
      text: "After everything that happened today, I can't help but feel incredibly happy.",
      class: 1,
    },
    {
      text: "This is just an ordinary ball, nothing special or noteworthy about it.",
      class: 0,
    },
    {
      text: "I can't stand it anymore, the more I think about it, the more I hate it.",
      class: -1,
    },
    {
      text: "The more I use it, the more I realize how much I love it.",
      class: 1,
    },
    {
      text: "This has turned out to be the best day of my life, and I couldn't be more thrilled.",
      class: 1,
    },
    {
      text: "From the very beginning, this has been the worst day I've ever experienced.",
      class: -1,
    },
    {
      text: "I didn't expect much, but now I find myself incredibly excited about the future.",
      class: 1,
    },
    {
      text: "Even though I tried to stay positive, I'm really disappointed with how things turned out.",
      class: -1,
    },
    {
      text: "Given all the circumstances, I'm actually feeling quite good about the situation.",
      class: 1,
    },
    {
      text: "The fear of power often leads people to make irrational decisions, which can be dangerous.",
      class: 0,
    },
    {
      text: "Despite my best efforts to remain optimistic, I'm feeling quite bad about everything.",
      class: -1,
    },
    {
      text: "Considering all the challenges, I'm feeling great about how I handled things.",
      class: 1,
    },
    {
      text: "No matter how hard I try to stay positive, I'm feeling terrible about the outcome.",
      class: -1,
    },
    {
      text: "Against all odds, I'm feeling absolutely awesome and ready to take on anything.",
      class: 1,
    },
    {
      text: "Even with all the support, I'm feeling awful and can't seem to shake this feeling.",
      class: -1,
    },
    {
      text: "With everything falling into place, I'm feeling fantastic about the future.",
      class: 1,
    },
    {
      text: "After carefully considering all aspects, I'm feeling okay with the decision.",
      class: 0,
    },
    {
      text: "Taking everything into account, I'm feeling fine about where we are right now.",
      class: 0,
    },
    {
      text: "All things considered, I'm feeling alright, though there's room for improvement.",
      class: 0,
    },
    {
      text: "Given the circumstances, I'm feeling good enough to proceed as planned.",
      class: 0,
    },
    {
      text: "Even with some setbacks, I'm feeling bad enough that it might affect my decisions.",
      class: 0,
    },
    {
      text: "In light of recent developments, I'm feeling great enough to continue pushing forward.",
      class: 0,
    },
    {
      text: "Despite the challenges, I'm feeling terrible enough to reconsider my options.",
      class: 0,
    },
  ]);
}

// Function to convert text to a vector using word2vec-like vectors
function textToVector(
  text: string,
  wordVectors: Record<string, number[]>
): number[][] {
  const words = text.toLowerCase().split(/\W+/);
  const vectors = words
    .map((word) => wordVectors[word])
    .filter((vector) => vector !== undefined);

  if (vectors.length === 0) {
    return Array(Object.values(wordVectors)[0].length).fill(0); // Return a zero vector if no known words are found
  }

  const summedVector = vectors.reduce(
    (sum, vec) => sum.map((v, i) => v + vec[i]),
    Array(vectors[0].length).fill(0) as number[]
  );
  return summedVector.map((v) => [v / vectors.length]); // Return the average vector
}

// Function to convert the inputs and predictions to CSV for 3D plotting
function convertToCSVFor3DPlot(
  inputs: number[][][],
  predictions: number[][][]
): string {
  const headers = ["x", "y", "z"]; // 'z' is the prediction or class
  const rows = inputs.map((input, index) => {
    const x = input[0][0]; // First word vector
    const y = input[1][0]; // Second word vector
    const z = predictions[index][0][0]; // Prediction probability for class 0 (or class 1 depending on the use case)
    return `${x},${y},${z}`;
  });
  return [headers.join(","), ...rows].join("\n");
}

const sentimentData = generateSentimentDataset();
const inputs = sentimentData.map((dataPoint) =>
  textToVector(dataPoint.text, wordVectors)
);
const targets = sentimentData.map((dataPoint) => {
  switch (dataPoint.class) {
    case -1:
      return [[1], [0], [0]]; // One-hot for Negative class (-1)
    case 0:
      return [[0], [1], [0]]; // One-hot for Neutral class (0)
    case 1:
      return [[0], [0], [1]]; // One-hot for Positive class (1)
    default:
      throw new Error("Unknown class value");
  }
});

const wordVectors_length = Object.values(wordVectors)[0].length;

// Initialize the Neural Network with the appropriate input size and three output neurons
const nn = new NeuralNetwork([wordVectors_length, 10, 3], {
  activationFunction: Activation.sigmoid,
  activationFunctionDerivative: Activation.sigmoidDerivative,
});
const csvData = convertToCSVFor3DPlot(
  inputs,
  sentimentData.map((d) => [[d.class]])
);
Bun.write("./testing.csv", csvData).then(console.log);

console.time("Sync Training Time");
nn.trainsync({ inputs, targets }, 500, 0.08);
console.timeEnd("Sync Training Time");

const model = nn.serialize();
Bun.write("./model.json", model).then(console.log);

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});


function start() {
  rl.question("Enter a sentence: ", (sentence) => {
    const vector = textToVector(sentence, wordVectors);
    const prediction = nn.predict(vector).flat();
    console.log("Prediction:", prediction);
    console.log("Class:", prediction.indexOf(Math.max(...prediction)) - 1);
    console.log("Negative:", prediction[0]);
    console.log("Neutral:", prediction[1]);
    console.log("Positive:", prediction[2]);
    // rl.close();
    start();
  });
}


// start();