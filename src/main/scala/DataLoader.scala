import scala.util.Random
import org.tensorflow.types.TInt32
import org.tensorflow.*
import org.tensorflow.ndarray.*
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.ModelType

class DataLoader(dataset: Dataset, batchSize: Int, shuffle: Boolean = true, dropLast: Boolean = true) {
  private var indices: Array[Int] = (0 until dataset.length).toArray

  // Shuffle the indices if needed
  if (shuffle) {
    indices = Random.shuffle(indices.toList).toArray
  }

  // Split the indices into batches
  private val batches: Array[Array[Int]] = indices.grouped(batchSize).toArray

  // Optionally drop the last batch if it's smaller than batchSize
  private val validBatches = if (dropLast && batches.last.length < batchSize) {
    batches.dropRight(1)
  } else {
    batches
  }

  private var currentBatch = 0

  // Get the next batch
  def nextBatch(): Option[Seq[(Tensor, Tensor)]] = {
    if (currentBatch >= validBatches.length) None
    else {
      val batchIndices = validBatches(currentBatch)
      currentBatch += 1
      Some(batchIndices.map(idx => dataset.getItem(idx)).toSeq)
    }
  }

  def hasNextBatch: Boolean = currentBatch < validBatches.length


}

def createDataloaderV1(txt: String, batchSize: Int = 4, maxLength: Int = 256, stride: Int = 128, shuffle: Boolean = true, dropLast: Boolean = true): DataLoader = {
  val registry = Encodings.newDefaultEncodingRegistry
  val tokenizer = registry.getEncodingForModel(ModelType.GPT_4)
  val dataset = new Dataset(txt, tokenizer, maxLength, stride)
  new DataLoader(dataset, batchSize, shuffle, dropLast)
}