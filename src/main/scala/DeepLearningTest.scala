import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{Encoding, IntArrayList, ModelType}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, RNNFormat}
import org.deeplearning4j.nn.conf.layers.{EmbeddingSequenceLayer, LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.activations.Activation
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.api.buffer.DataType

import scala.collection.mutable.ArrayBuffer

object TokenEmbeddingExample {

  def main(args: Array[String]): Unit = {

    val filePath = "C:\\Users\\DELL\\Downloads\\Verdict.txt"

    val registry = Encodings.newDefaultEncodingRegistry
    // Initialize tokenizer
    val tokenizer: Encoding = registry.getEncodingForModel(ModelType.GPT_4)

    // Load and tokenize text
   // val text = "This is a sample for test. Sample for project. This is a test."
    val text: String = scala.io.Source.fromFile(filePath).mkString
    val (remappedTokenIds, tokenIdToIndex, indexToTokenId,vocabSize) = tokenizeAndMapTokens(text, tokenizer)

    println(remappedTokenIds.mkString(", "))

    // Parameters
    val windowSize = 3
    val stride = 2

    // Generate input-target pairs using sliding windows with the remapped token IDs
    val pairs = generateInputTargetPairs(remappedTokenIds, windowSize, stride)

    //val vocabSize = _

    // Prepare training data
    val (inputArray, targetArray) = createTrainingData(pairs)

    // Verify data shapes and types
    println(s"Input Array Data Type: ${inputArray.dataType()}")
    println(s"Input Array Shape: ${inputArray.shape().mkString(", ")}")
    println(s"Target Array Data Type: ${targetArray.dataType()}")
    println(s"Target Array Shape: ${targetArray.shape().mkString(", ")}")

    // Build and train the model
    val embeddingSize = 2
    val model = buildModel(vocabSize, embeddingSize)
    val batchSize = 128
    val numEpochs = 5
    trainModel(model, inputArray, targetArray, batchSize, numEpochs)

    // Extract and output embeddings in the order of the text
    val embeddings = extractEmbeddingsInOrder(model, remappedTokenIds, indexToTokenId, tokenizer)
    embeddings.foreach { case (word, tokenId, embeddingVector) =>
      println(s"Word: '$word', Token ID: $tokenId, Embedding: ${embeddingVector}")
    }
  }

  /** Tokenize the text using JTokkit and map tokens to new indices starting from 0 */
  def tokenizeAndMapTokens(text: String, tokenizer: Encoding): (Array[Int], Map[Int, Int], Map[Int, Int],Int) = {
    val wordPattern = "[a-zA-Z0-9']+".r
    val tokenBuffer = ArrayBuffer[Int]() // Use ArrayBuffer to accumulate tokens
    val tokenSet = ArrayBuffer[Int]() // To ensure that we map tokens without removing duplicates
    val tokens = wordPattern.findAllIn(text)

    tokens.foreach { token =>
      val encodedList = tokenizer.encode(token.strip())
      (0 until encodedList.size()).foreach { i =>
        val tokenId = encodedList.get(i)
        tokenBuffer += tokenId
        if (!tokenSet.contains(tokenId)) tokenSet += tokenId // Add unique token to set
      }
    }

    // Create a mapping from token IDs to indices starting from 0, preserving order
    val tokenIdToIndex = tokenSet.zipWithIndex.toMap
    val indexToTokenId = tokenIdToIndex.map(_.swap)

    // Remap the original tokenBuffer using tokenIdToIndex mapping
    val remappedTokenIds = tokenBuffer.map(tokenIdToIndex).toArray

    (remappedTokenIds, tokenIdToIndex, indexToTokenId,tokenSet.size)
  }

  /** Generate input-target pairs using sliding windows */
  def generateInputTargetPairs(tokenIds: Array[Int], windowSize: Int, stride: Int): Seq[(Array[Int], Array[Int])] = {
    val inputWindows = tokenIds.sliding(windowSize, stride).toSeq
    val targetWindows = tokenIds.slice(1, tokenIds.length).sliding(windowSize, stride).toSeq

    inputWindows.zip(targetWindows).filter { case (input, target) =>
      input.length == windowSize && target.length == windowSize
    }
  }

  /** Create training data arrays */
  def createTrainingData(pairs: Seq[(Array[Int], Array[Int])]): (INDArray, INDArray) = {
    val numSamples = pairs.length
    val sequenceLength = pairs.head._1.length

    // Inputs: [numSamples, sequenceLength]
    val inputArray = Nd4j.create(DataType.INT, numSamples, sequenceLength)
    // Targets: [numSamples, 1, sequenceLength]
    val targetArray = Nd4j.create(DataType.INT, numSamples, 1, sequenceLength)

    pairs.zipWithIndex.foreach { case ((inputIndices, targetIndices), idx) =>
      // Input data: sequence of token indices
      val inputSequence = Nd4j.createFromArray(inputIndices.map(_.toInt): _*).castTo(DataType.INT)
      inputArray.putRow(idx, inputSequence)

      // Target data: integer labels reshaped to [1, sequenceLength]
      val targetSequence = Nd4j.createFromArray(targetIndices.map(_.toInt): _*).castTo(DataType.INT).reshape(1, sequenceLength)
      targetArray.putRow(idx, targetSequence)
    }

    (inputArray, targetArray)
  }

  /** Build the neural network model */
  def buildModel(vocabSize: Int, embeddingSize: Int): MultiLayerNetwork = {
    val conf = new NeuralNetConfiguration.Builder()
      .updater(new Adam(0.001))
      .list()
      .layer(0, new EmbeddingSequenceLayer.Builder()
        .nIn(vocabSize)
        .nOut(embeddingSize)
        .build())
      .layer(1, new LSTM.Builder()
        .nIn(embeddingSize)
        .nOut(128)
        .activation(Activation.TANH)
        .build())
      .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
        .activation(Activation.SOFTMAX)
        .nIn(128)
        .nOut(vocabSize)
        .dataFormat(RNNFormat.NCW) // Ensure the data format matches the labels
        .build())
      .setInputType(InputType.recurrent(1))
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model
  }

  /** Train the neural network model */
  def trainModel(model: MultiLayerNetwork, inputArray: INDArray, targetArray: INDArray, batchSize: Int, numEpochs: Int): Unit = {
    val dataSet = new DataSet(inputArray, targetArray)
    val dataSetIterator = new ListDataSetIterator(dataSet.asList(), batchSize)
    model.setListeners(new ScoreIterationListener(10))

    (1 to numEpochs).foreach { epoch =>
      dataSetIterator.reset()
      model.fit(dataSetIterator)
      println(s"Epoch $epoch complete")
    }
  }

  /** Extract and output embeddings in the order of the text */
  def extractEmbeddingsInOrder(model: MultiLayerNetwork, remappedTokenIds: Array[Int], indexToTokenId: Map[Int, Int], tokenizer: Encoding): Seq[(String, Int, INDArray)] = {
    val embeddingWeights = model.getLayer(0).getParam("W") // Shape: [vocabSize, embeddingSize]
    remappedTokenIds.map { remappedTokenId =>
      val embeddingVector = embeddingWeights.getRow(remappedTokenId)
      val originalTokenId = indexToTokenId(remappedTokenId)
      // Decode the single token ID
      val list=new IntArrayList()
      list.add(originalTokenId)
      val word = tokenizer.decode(list)
      (word, originalTokenId, embeddingVector)
    }
  }

}
