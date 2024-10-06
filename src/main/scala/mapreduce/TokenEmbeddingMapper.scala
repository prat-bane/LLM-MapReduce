package mapreduce

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{Encoding, IntArrayList, ModelType}
import com.typesafe.config.ConfigFactory
import org.apache.hadoop.io.Text
import org.apache.hadoop.mapreduce.Mapper
import org.slf4j.LoggerFactory
import utility.{ModelUtil, SlidingWindowUtil}



class TokenEmbeddingMapper extends Mapper[Text, Text, Text, Text] {

  // Initialize the logger
  private val logger = LoggerFactory.getLogger(classOf[TokenEmbeddingMapper])

  private val registry = Encodings.newDefaultEncodingRegistry
  // Initialize tokenizer
  private val tokenizer: Encoding = registry.getEncodingForModel(ModelType.GPT_4)
  private val config = ConfigFactory.load()
  private val windowSize: Int = config.getInt("app.embedding-config.window-size")
  private val stride: Int = config.getInt("app.embedding-config.stride")
  private val embeddingSize: Int = config.getInt("app.embedding-config.embedding-size")

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

      // Map tokens to indices so that we don't get ArrayIndexOutofBoundExceptions
      val uniqueTokens = tokens.distinct
      val tokenToIndex = uniqueTokens.zipWithIndex.toMap
      val vocabSize = uniqueTokens.size
      logger.info(s"Unique tokens: $vocabSize")

      // Map tokens to indices
      val tokenIndices = tokens.map(tokenToIndex)

      // Generate input-target pairs
      val inputTargetPairs = SlidingWindowUtil.generateInputTargetPairs(tokenIndices.toArray, windowSize, stride)
      logger.info(s"Generated ${inputTargetPairs.length} input-target pairs")
      if (inputTargetPairs.nonEmpty) {
        // Create training data
        val (inputArray, targetArray) = SlidingWindowUtil.createTrainingData(inputTargetPairs)

        // Build and train the model
        val model = ModelUtil.buildModel(vocabSize, embeddingSize)
        logger.info("Starting model training")
        ModelUtil.trainModel(model, inputArray, targetArray)
        logger.info("Model training completed")

        // Extract embeddings
        val embeddings = ModelUtil.extractEmbeddings(model)
        logger.info("Extracted embeddings")

        // Emit embeddings: (token, embedding vector)
        uniqueTokens.foreach { token =>
          val index = tokenToIndex(token)
          val embeddingVector = embeddings(index)
          val embeddingString = embeddingVector.toDoubleVector.mkString(",")
          val tokenList= new IntArrayList(1)
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

}
