package utility

import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.{LoggerFactory}

object SlidingWindowUtil {

  private val logger = LoggerFactory.getLogger(SlidingWindowUtil.getClass)

  def generateInputTargetPairs(tokenIds: Array[Int], windowSize: Int, stride: Int): Seq[(Array[Int], Array[Int])] = {
    logger.info(s"Generating input-target pairs with windowSize=$windowSize and stride=$stride")

    // Log the length of the tokenIds array
    logger.debug(s"Token ID array length: ${tokenIds.length}")

    val inputWindows = tokenIds.sliding(windowSize, stride).toSeq
    val targetWindows = tokenIds.slice(1, tokenIds.length).sliding(windowSize, stride).toSeq

    val inputTargetPairs = inputWindows.zip(targetWindows).filter { case (input, target) =>
      input.length == windowSize && target.length == windowSize
    }

    // Log the number of input-target pairs generated
    logger.info(s"Generated ${inputTargetPairs.length} input-target pairs")
    inputTargetPairs
  }

  /** Creates training data arrays from input-target pairs */
  def createTrainingData(pairs: Seq[(Array[Int], Array[Int])]): (INDArray, INDArray) = {
    logger.info(s"Creating training data from ${pairs.length} input-target pairs")

    val numSamples = pairs.length
    val sequenceLength = pairs.head._1.length

    // Log the sequence length and number of samples
    logger.debug(s"Number of samples: $numSamples, Sequence length: $sequenceLength")

    // Inputs: [numSamples, sequenceLength]
    val inputArray = Nd4j.create(DataType.INT, numSamples, sequenceLength)
    // Targets: [numSamples, 1, sequenceLength]
    val targetArray = Nd4j.create(DataType.INT, numSamples, 1, sequenceLength)

    pairs.zipWithIndex.foreach { case ((inputIndices, targetIndices), idx) =>
      // Log each sample being added
      logger.debug(s"Adding sample $idx: input=${inputIndices.mkString(",")} target=${targetIndices.mkString(",")}")

      // Input data: sequence of token indices
      val inputSequence = Nd4j.createFromArray(inputIndices.map(_.toInt): _*).castTo(DataType.INT)
      inputArray.putRow(idx, inputSequence)

      // Target data: integer labels reshaped to [1, sequenceLength]
      val targetSequence = Nd4j.createFromArray(targetIndices.map(_.toInt): _*).castTo(DataType.INT).reshape(1, sequenceLength)
      targetArray.putRow(idx, targetSequence)
    }

    logger.info("Training data creation completed")
    (inputArray, targetArray)
  }
}
