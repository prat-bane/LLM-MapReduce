package mapreduce

import org.apache.hadoop.io._
import org.apache.hadoop.mapreduce._
import scala.collection.JavaConverters._
import org.nd4j.linalg.factory.Nd4j
import org.apache.log4j.Logger

class TokenEmbeddingReducer extends Reducer[Text, Text, Text, Text] {

  // Initialize the logger
  private val logger = Logger.getLogger(classOf[TokenEmbeddingReducer])

  override def reduce(
                       key: Text,
                       values: java.lang.Iterable[Text],
                       context: Reducer[Text, Text, Text, Text]#Context
                     ): Unit = {
    try {
      logger.info(s"Reducing embeddings for token: ${key.toString}")


      val embeddings = values.asScala.map { value =>
        val vector = value.toString.trim.split(",").map(_.toDouble)
        Nd4j.create(vector)
      }.toSeq

      if (embeddings.nonEmpty) {
        // Sum embeddings
        val sumEmbedding = embeddings.reduce(_.add(_))

        // Average embeddings
        val avgEmbedding = sumEmbedding.div(embeddings.size.toDouble)

        val embeddingString = avgEmbedding.toDoubleVector.mkString(",")
        context.write(key, new Text(embeddingString))
        logger.info(s"Emitted averaged embedding for token: ${key.toString}")
      } else {
        logger.warn(s"No embeddings found for token: ${key.toString}")
      }
    } catch {
      case e: Exception =>
        logger.error(s"Error reducing embeddings for token: ${key.toString}", e)
        throw e
    }
  }
}
