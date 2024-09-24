import org.tensorflow.*
import org.tensorflow.ndarray.*
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import org.tensorflow.types.TInt32

import java.nio.IntBuffer
import scala.collection.mutable.ArrayBuffer

class Dataset(txt: String, tokenizer: Encoding, maxLength: Int, stride: Int) {

  private val inputIds = ArrayBuffer[Tensor]()
  private val targetIds = ArrayBuffer[Tensor]()

  // Tokenizing the input text using JTokkit and converting IntArrayList to a Scala array
  val tokenIds: Array[Int] = tokenizer.encode(txt).toArray
  println(tokenIds.length)

  // Sliding window creation for input and target sequences
  val tokenPairs = tokenIds.sliding(maxLength + 1, stride).toArray // Creates windows of size (maxLength + 1)

  for (pair <- tokenPairs) {
    val inputChunk = pair.take(maxLength) // Input: first maxLength tokens
    val targetChunk = pair.drop(1).take(maxLength)// Target: next maxLength tokens
    println(s"Input Chunk: ${inputChunk.mkString(", ")}")
    println(s"Target Chunk: ${targetChunk.mkString(", ")}")


    // Create NdArrays for input and target
    val inputNdArray = NdArrays.ofInts(Shape.of(inputChunk.length))
    inputChunk.zipWithIndex.foreach { case (value, idx) => inputNdArray.setInt(value, idx) }

    val targetNdArray = NdArrays.ofInts(Shape.of(targetChunk.length))
    targetChunk.zipWithIndex.foreach { case (value, idx) => targetNdArray.setInt(value, idx) }
    
    // Convert NdArrays to Tensors
    inputIds += TInt32.tensorOf(inputNdArray.asInstanceOf[NdArray[Integer]])
    targetIds += TInt32.tensorOf(targetNdArray.asInstanceOf[NdArray[Integer]])
  }

  // Method to get the length of the dataset
  def length: Int = inputIds.length

  // Method to get an item at a specific index
  def getItem(idx: Int): (Tensor, Tensor) = {
    (inputIds(idx), targetIds(idx))
  }

  def printTensors(): Unit = {
    println("Printing Tensors:")

    // Loop through all input tensors
    for ((inputTensor, targetTensor) <- inputIds.zip(targetIds)) {
      println("Input Tensor:")
      printTensorContent(inputTensor)

      println("Target Tensor:")
      printTensorContent(targetTensor)
    }
  }

  // Helper method to print the content of a tensor
  def printTensorContent(tensor: Tensor): Unit = {
    val intBuffer= tensor.asRawTensor().data().asInts()

    // Iterate through the buffer and print each value
    for (i <- 0 until intBuffer.size().toInt) {
      println(intBuffer.getInt(i))
    }
  }

  // Helper method to load the dataset in batches (if needed)
 
}

object Dataset {
  def apply(txt: String, tokenizer: Encoding, maxLength: Int, stride: Int): Dataset = {
    new Dataset(txt, tokenizer, maxLength, stride)
  }
}