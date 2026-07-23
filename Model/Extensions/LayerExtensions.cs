namespace SimpleTransformer.Model.Extensions
{
    public static class LayerExtensions
    {
        public static List<ILayer> ReverseLayers(List<ILayer> layers)
        {
            layers.Reverse();
            return layers;
        }
    }
}