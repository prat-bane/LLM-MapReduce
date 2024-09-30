package mapreduce

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{Encoding, IntArrayList, ModelType}
import org.apache.hadoop.io.*
import org.apache.hadoop.mapreduce.*

import scala.collection.mutable
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, RNNFormat}
import org.deeplearning4j.nn.conf.layers.{EmbeddingSequenceLayer, LSTM, RnnOutputLayer}
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.activations.Activation
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.apache.log4j.Logger

class TokenEmbeddingMapper extends Mapper[Text, Text, Text, Text] {

  // Initialize the logger
  private val logger = Logger.getLogger(classOf[TokenEmbeddingMapper])

  val registry = Encodings.newDefaultEncodingRegistry
  // Initialize tokenizer
  val tokenizer: Encoding = registry.getEncodingForModel(ModelType.GPT_4)

  override def map(
                    key: Text,
                    value: Text,
                    context: Mapper[Text, Text, Text, Text]#Context
                  ): Unit = {
    try {
      logger.info(s"Processing file: ${key.toString}")

      // Process lines individually
      val lines = value.toString.linesIterator

      // Map lines to tokens (numbers)
      val tokens = lines.map(_.trim).filter(_.nonEmpty).toSeq
      logger.info(s"Total tokens read: ${tokens.length}")

      // Map tokens to indices (considering duplicates)
      val uniqueTokens = tokens.distinct
      val tokenToIndex = uniqueTokens.zipWithIndex.toMap
      val indexToToken = tokenToIndex.map(_.swap)
      val vocabSize = uniqueTokens.size
      logger.info(s"Unique tokens: $vocabSize")

      // Map tokens to indices
      val tokenIndices = tokens.map(tokenToIndex)

      // Generate input-target pairs using sliding windows
      val windowSize = 3
      val stride = 1

      val inputTargetPairs = generateInputTargetPairs(tokenIndices.toArray, windowSize, stride)
      logger.info(s"Generated ${inputTargetPairs.length} input-target pairs")

      if (inputTargetPairs.nonEmpty) {
        // Create training data
        val (inputArray, targetArray) = createTrainingData(inputTargetPairs)

        // Build and train the model
        val embeddingSize = 100
        val model = buildModel(vocabSize, embeddingSize)
        logger.info("Starting model training")
        trainModel(model, inputArray, targetArray)
        logger.info("Model training completed")

        // Extract embeddings
        val embeddings = extractEmbeddings(model)
        logger.info("Extracted embeddings")

        // Emit embeddings: (token, embedding vector)
        uniqueTokens.foreach { token =>
          val index = tokenToIndex(token)
          val embeddingVector = embeddings(index)
          val embeddingString = embeddingVector.toDoubleVector.mkString(",")
          val tokenList=IntArrayList(1)
          tokenList.add(token.toInt)
          val word= tokenizer.decode(tokenList)
          val keyOut = s"$word $token"
          context.write(new Text(keyOut), new Text(embeddingString))
        }
        logger.info("Emitted embeddings for all tokens")
      } else {
        logger.warn("No input-target pairs generated; skipping model training")
      }
    } catch {
      case e: Exception =>
        logger.error(s"Error processing file: ${key.toString}", e)
        throw e
    }
  }

  /** Generate input-target pairs using sliding windows */
  def generateInputTargetPairs(tokenIds: Array[Int], windowSize: Int, stride: Int): Seq[(Array[Int], Array[Int])] = {
    val inputWindows = tokenIds.sliding(windowSize, stride).toSeq
    val targetWindows = tokenIds.slice(1, tokenIds.length).sliding(windowSize, stride).toSeq

    inputWindows.zip(targetWindows).filter { case (input, target) =>
      input.length == windowSize && target.length == windowSize
    }
  }

  /** Creates training data arrays from input-target pairs */
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

  /** Builds the neural network model */
  def buildModel(vocabSize: Int, embeddingSize: Int): MultiLayerNetwork = {
    val conf = new NeuralNetConfiguration.Builder()
      .updater(new org.nd4j.linalg.learning.config.Adam(0.001))
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
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model
  }

  /** Trains the neural network model */
  def trainModel(model: MultiLayerNetwork, inputArray: INDArray, targetArray: INDArray): Unit = {
    val dataSet = new DataSet(inputArray, targetArray)
    val batchSize = 128
    val dataSetIterator = new ListDataSetIterator(dataSet.asList(), batchSize)
    model.setListeners(new ScoreIterationListener(10))

    (1 to 5).foreach { epoch =>
      dataSetIterator.reset()
      model.fit(dataSetIterator)
      logger.info(s"Epoch $epoch completed")
    }
  }

  /** Extracts embeddings from the model */
  def extractEmbeddings(model: MultiLayerNetwork): Map[Int, INDArray] = {
    val embeddingWeights = model.getLayer(0).getParam("W") // Shape: [vocabSize, embeddingSize]
    val vocabSize = embeddingWeights.rows()
    (0 until vocabSize).map { i =>
      val embeddingVector = embeddingWeights.getRow(i).dup()
      i -> embeddingVector
    }.toMap
  }
}
