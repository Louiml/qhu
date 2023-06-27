vocabSize = 3285;
embeddingDim = 128;
numHeads = 4;
numLayers = 2;

formula = HoldForm[   GPT[vocabSize_, embeddingDim_, numHeads_, numLayers_] := Module[
     {embedding, posEncoder, transformerEncoder, fc},
     embedding = Embedding[vocabSize, embeddingDim];
     posEncoder = PositionalEncoding[embeddingDim];
     transformerEncoder = TransformerEncoder[embeddingDim, numHeads, numLayers];
     fc = LinearLayer[vocabSize];
     
     GPT[input_] := Module[
       {embedded, encoded, output},
       embedded = embedding[input];
       encoded = posEncoder[embedded];
       output = transformerEncoder[encoded];
       fc[output]
     ];
     
     GPT
   ];
];

DisplayForm[formula]
