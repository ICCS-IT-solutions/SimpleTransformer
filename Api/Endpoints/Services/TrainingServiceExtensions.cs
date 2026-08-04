namespace SimpleTransformer.Api.Endpoints.Services
{
    public static class TrainingServiceExtensions
    {
        public static void ShuffleInPlace<T>(this IList<T> list, Random random)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = random.Next(n + 1);
                (list[k], list[n]) = (list[n], list[k]);
            }
        }
    }
}