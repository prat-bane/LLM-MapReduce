package utility

package utility

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.mockito.{ArgumentMatchers, MockitoSugar}
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.funsuite.AnyFunSuite
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet

class ModelUtilTest extends AnyFunSuite with MockitoSugar {

  test("buildModel should return a correctly configured MultiLayerNetwork") {
    val vocabSize = 100
    val embeddingSize = 50

    // Build the model using ModelUtil
    val model = ModelUtil.buildModel(vocabSize, embeddingSize)

    // Verify the model has the expected structure
    val conf = model.getLayerWiseConfigurations
    assert(conf.getConf(0).getLayer.getClass.getSimpleName == "EmbeddingSequenceLayer", "Layer 0 should be an EmbeddingSequenceLayer")
    assert(conf.getConf(1).getLayer.getClass.getSimpleName == "LSTM", "Layer 1 should be an LSTM")
    assert(conf.getConf(2).getLayer.getClass.getSimpleName == "RnnOutputLayer", "Layer 2 should be an RnnOutputLayer")

    // Verify that the first layer has the correct input and output size
    val embeddingLayerConf = conf.getConf(0).getLayer.asInstanceOf[org.deeplearning4j.nn.conf.layers.EmbeddingSequenceLayer]
    assert(embeddingLayerConf.getNIn == vocabSize, s"Expected input size of vocabSize $vocabSize")
    assert(embeddingLayerConf.getNOut == embeddingSize, s"Expected output size of embeddingSize $embeddingSize")

  }

  test("trainModel should train the model for the specified number of epochs") {
    // Mock the model
    val model = mock[MultiLayerNetwork]
    val inputArray: INDArray = Nd4j.zeros(10, 3)
    val targetArray: INDArray = Nd4j.zeros(10, 3)

    // Call the trainModel function
    ModelUtil.trainModel(model, inputArray, targetArray)

    // Verify that the model's fit method was called the correct number of times (epochs)
    verify(model, times(ModelUtil.epochs)).fit(ArgumentMatchers.any(classOf[ListDataSetIterator[DataSet]]))
  }

  test("extractEmbeddings should return the correct embedding map") {
    // Create a mock model and layer
    val model = mock[MultiLayerNetwork]
    val layer = mock[org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer]

    val vocabSize = 100
    val embeddingSize = 50

    // Mock the embedding layer parameters
    val embeddingWeights: INDArray = Nd4j.rand(vocabSize, embeddingSize)

    // Mock the getLayer(0) and getParam("W") methods
    when(model.getLayer(0)).thenReturn(layer)
    when(layer.getParam("W")).thenReturn(embeddingWeights)

    // Call the extractEmbeddings function
    val embeddings = ModelUtil.extractEmbeddings(model)

    // Verify the size of the embeddings map
    assert(embeddings.size == vocabSize, s"Expected vocabSize $vocabSize")

    // Verify the embedding for a specific word
    val firstEmbedding = embeddings(0)
    assert(firstEmbedding.shape().sameElements(Array(embeddingSize)), s"Expected embedding size $embeddingSize")

    // Check that the first embedding vector matches the expected values
    assert(firstEmbedding.equals(embeddingWeights.getRow(0)), "Embedding vector should match the row from the embedding matrix")
  }

}

