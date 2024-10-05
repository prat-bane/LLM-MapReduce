package utility

import com.typesafe.config.ConfigFactory
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, RNNFormat}
import org.deeplearning4j.nn.conf.layers.{EmbeddingSequenceLayer, LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

object ModelUtil {

  private val config = ConfigFactory.load()
  private val lstmLayerSize: Int = config.getInt("app.embedding-config.lstm-layer-size")
  val learningRate: Double = config.getDouble("app.embedding-config.learning-rate")
  val epochs: Int = config.getInt("app.embedding-config.epochs")
  private val batchSize: Int = config.getInt("app.embedding-config.batch-size")
  private val logger = LoggerFactory.getLogger(ModelUtil.getClass)

  def buildModel(vocabSize: Int, embeddingSize: Int): MultiLayerNetwork = {
    logger.info(s"Building model with vocabSize: $vocabSize, embeddingSize: $embeddingSize")
    val conf = new NeuralNetConfiguration.Builder()
      .updater(new Adam(learningRate))
      .list()
      .layer(0, new EmbeddingSequenceLayer.Builder()
        .nIn(vocabSize)
        .nOut(embeddingSize)
        .build())
      .layer(1, new LSTM.Builder()
        .nIn(embeddingSize)
        .nOut(lstmLayerSize) // Set from config
        .activation(Activation.TANH)
        .build())
      .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
        .activation(Activation.SOFTMAX)
        .nIn(lstmLayerSize) // Set from config
        .nOut(vocabSize)
        .dataFormat(RNNFormat.NCW)
        .build())
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    logger.info("Model initialized successfully")
    model
  }

  def trainModel(model: MultiLayerNetwork, inputArray: INDArray, targetArray: INDArray): Unit = {
    logger.info("Starting model training...")
    val dataSet = new DataSet(inputArray, targetArray)
    val dataSetIterator = new ListDataSetIterator(dataSet.asList(), batchSize) // Batch size from config
    model.setListeners(new ScoreIterationListener(10))

    (1 to epochs).foreach { epoch =>
      dataSetIterator.reset()
      model.fit(dataSetIterator)
      logger.info(s"Epoch $epoch completed")
    }

    logger.info("Model training completed")
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
