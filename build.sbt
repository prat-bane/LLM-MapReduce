ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.12"

lazy val root = (project in file("."))
  .settings(
    name := "ScalaRest"
  )

assembly / assemblyMergeStrategy := {
  case PathList("META-INF", "LICENSE") => MergeStrategy.discard
  case PathList("META-INF", "License") => MergeStrategy.discard
  case PathList("META-INF", "LICENSE.txt") => MergeStrategy.discard
  case PathList("META-INF", "License.txt") => MergeStrategy.discard
  case PathList("META-INF", "license") => MergeStrategy.discard
  case PathList("META-INF", "license.txt") => MergeStrategy.discard
  case PathList("META-INF", xs @ _*) =>
    xs match {
      case "MANIFEST.MF" :: Nil => MergeStrategy.discard
      case "services" :: _ => MergeStrategy.concat
      case _ => MergeStrategy.discard
    }
  case "reference.conf" => MergeStrategy.concat
  case x if x.endsWith(".proto") => MergeStrategy.rename
  case x if x.contains("hadoop") => MergeStrategy.first
  case _ => MergeStrategy.first
}

fork := true

javaOptions += "-Djavacpp.platform=windows-x86_64"

libraryDependencies ++= Seq(
  "com.knuddels" % "jtokkit" % "1.1.0",
  "org.apache.hadoop" % "hadoop-common" % "3.3.4",
  "org.apache.hadoop" % "hadoop-mapreduce-client-core" % "3.3.4",
  "org.apache.hadoop" % "hadoop-mapreduce-client-jobclient" % "3.3.4",
  "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-beta7",
  "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-beta7",
  "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta7",
  "org.slf4j" % "slf4j-simple" % "2.0.13",
  "com.typesafe" % "config" % "1.4.3",
  "org.mockito" %% "mockito-scala" % "1.17.14" % Test,
  "org.scalatest" %% "scalatest" % "3.2.17" % Test
)






excludeDependencies ++= Seq(
  //ExclusionRule(organization = "org.bytedeco", name = ""),
  ExclusionRule(organization = "com.twelvemonkeys.common"),
  ExclusionRule(organization = "com.twelvemonkeys.imageio"),
  ExclusionRule(organization = "org.bytedeco",name= "javacpp:1.5.6:android-arm64"),
  ExclusionRule(organization = "org.bytedeco", name= "javacpp:1.5.6:android-arm"),

)