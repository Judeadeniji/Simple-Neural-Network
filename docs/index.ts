import negativeWords from "./negative-words.txt";
import positiveWords from "./positive-words.txt";

const negativeWordSet = new Set<string>(negativeWords.split("\n"));
const positiveWordSet = new Set<string>(positiveWords.split("\n"));

const dataPoints: {
  text: string;
  class: number;
}[] = [];

negativeWordSet.forEach((word) => {
  dataPoints.push({ text: word, class: -1 });
});

positiveWordSet.forEach((word) => {
  dataPoints.push({ text: word, class: 1 });
});


export {
    dataPoints,
}