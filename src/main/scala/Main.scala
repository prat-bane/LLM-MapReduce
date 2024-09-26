import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{EncodingType, ModelType}
import org.tensorflow.*
import org.tensorflow.ndarray.*
import org.tensorflow.ndarray.buffer.DataBuffers
import org.tensorflow.proto.framework.{MetaGraphDef, SignatureDef}
import org.tensorflow.types.TInt32
import scala.collection.JavaConverters.asScalaIteratorConverter

import java.nio.file.{Files, Paths}
import java.util
import scala.io.Source

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j


object Main {

  @main def hello(): Unit =


    val maxHeapSize = Runtime.getRuntime.maxMemory() / (1024 * 1024)
    println(s"Max Heap Size (MB): $maxHeapSize")

    val filePath = "C:\\Users\\DELL\\Downloads\\Verdict.txt"

    val registry = Encodings.newDefaultEncodingRegistry
    val encoding = registry.getEncodingForModel(ModelType.GPT_4)

    // Reading the file content into a String
    val fileContent: String = Files.readString(Paths.get(filePath))
 //   val deepLearningTest = new DeepLearningTest(fileContent, encoding, 4, 1)
    val txt = "This is a sample text for embedding layer testing.Model should run correctly"

      // Define sliding window parameters
      val maxLength: Int = 10
      val stride: Int = 1

      // Instantiate the DeepLearningTest class
     //val deepLearningTest = new DeepLearningTest(fileContent, encoding, maxLength, stride)

      // Initialize data preparation and model configuration
    /*  deepLearningTest.initialize()

      // Train the model
      deepLearningTest.trainModel()

      // (Optional) Print learned embeddings
      deepLearningTest.printWordTokenEmbedding()
      deepLearningTest.exportEmbeddingsToCSV("D:\\IdeaProjects\\ScalaRest\\src\\main\\resources\\output.csv")*/
   // println(fileContent)

   // val encoded = encoding.encode("Hello babe")
   /* for (i <- 0 until encoded.size()) {
    //  println(encoded.get(i))
    }*/


    // Step 3: Convert Array[Int] to NdArray[Int] using StdArrays.ndCopyOf
   // val ndArray: NdArray[Int] = StdArrays.ndCopyOf(intArray)

   /* val intArray: Array[Int] = encoded.toArray();
    val dataBuffer = DataBuffers.of(intArray, false, false)

    // Step 4: Create an NdArray[Int] with the desired shape
    val shape = Shape.of(intArray.length) // For a 1D array
    val ndArray: IntNdArray = NdArrays.ofInts(shape)

    // Step 5: Populate the NdArray with data from the DataBuffer
    ndArray.copyFrom(dataBuffer)

    val tensor = TInt32.tensorOf(ndArray)
    //val decoded = encoding.decode(encoded)

    println(tensor)*/
   // val txt = "This is an example text to tokenize and generate tensors from."

    // Initialize the tokenizer (for example, using GPT-2 encoding)
   // val tokenizer: Encoding = Encodings.forModel("gpt2")

    // Define maxLength and stride for the sliding window


    // Create the Dataset instance
 //   val dataset = Dataset(fileContent, encoding, maxLength, stride)

 //   // Output the length of the dataset
 //   println(s"Dataset length: ${dataset.length}")
 //   dataset.printTensors()
    // Get and print the first item (input tensor and target tensor)
    /*if (dataset.length > 0) {
      val (inputTensor, targetTensor) = dataset.getItem(0)
        // Access the NdArray from the Tensor
      val ndArray: NdArray[Int] = inputTensor.asInstanceOf[NdArray[Int]]

      // Iterate and print the content of the tensor
      ndArray.scalars().forEach(value => println(value.get()))
      println(inputTensor.toString)
      println(s"Input Tensor: $inputTensor")
      println(s"Target Tensor: $targetTensor")*/

   /* val filePath = "C:\\Users\\DELL\\Downloads\\Verdict.txt"
    val rawText = Source.fromFile(filePath).mkString

    // Create a DataLoader with batchSize = 1, maxLength = 4, and stride = 1
    val dataLoader = createDataloaderV1(
      rawText, batchSize = 1, maxLength = 4, stride = 1, shuffle = false
    )
    val batch = dataLoader.nextBatch().get
    batch.foreach { case (inputTensor, targetTensor) =>
      println(s"Input Tensor: $inputTensor")
      println(s"Target Tensor: $targetTensor")
    }*/

    // Process batches from the DataLoader
   /* while (dataLoader.hasNextBatch) {
      val batch = dataLoader.nextBatch().get
      println("Processing Batch:")
      batch.foreach { case (inputTensor, targetTensor) =>
        println(s"Input Tensor: $inputTensor")
        println(s"Target Tensor: $targetTensor")
      }
    }*/

   /*val path="D:\\bert-tensorflow2-bert-en-uncased-l-10-h-128-a-2-v2"
   val model= SavedModelBundle.load(path)
  println("Model loaded successfully!")

  val inputWordIds = Array(Array(101, 2009, 2003, 102)) // Example tokenized input
  val inputMask = Array(Array(1, 1, 1, 1)) // Mask indicating valid tokens
  val inputTypeIds = Array(Array(0, 0, 0, 0)) // Segment IDs

  // Prepare tensors
  val inputWordIdsTensor: Tensor = createTensor(inputWordIds)
  val inputMaskTensor: Tensor = createTensor(inputMask)
  val inputTypeIdsTensor: Tensor = createTensor(inputTypeIds)

  // Run inference
  val outputTensors: java.util.List[Tensor] = model.session()
    .runner()
    .feed("serving_default_input_word_ids:0", inputWordIdsTensor)
    .feed("serving_default_input_mask:0", inputMaskTensor)
    .feed("serving_default_input_type_ids:0", inputTypeIdsTensor)
    .fetch("StatefulPartitionedCall:0") // Get the first output tensor
    .run()
  println("Size" + outputTensors.size())
  // Extract and print the output
  val outputTensor = outputTensors.get(0)
  println("Output tensor: " + outputTensor)

  val intBuffer = outputTensor.asRawTensor().data().asInts()

  // Iterate through the buffer and print each value
  for (i <- 0 until intBuffer.size().toInt) {
    println(intBuffer.getInt(i))
  }

  // Cleanup
  model.close()
  inputWordIdsTensor.close()
  inputMaskTensor.close()
  inputTypeIdsTensor.close()
  outputTensor.close()*/


}
def createTensor(data: Array[Array[Int]]): Tensor = {
  // Flatten the input data
  val flattenedData = data.flatten

  // Create a tensor using the flattened data
  val tensor: Tensor = TInt32.tensorOf(NdArrays.ofInts(Shape.of(data.length, data(0).length)))

  // Fill the tensor with data
  val intBuffer = tensor.asRawTensor().data().asInts()  // Access the underlying buffer for ints
  for (i <- flattenedData.indices) {
    intBuffer.setInt(flattenedData(i), i)
  }

  tensor  // Return the populated tensor
}


