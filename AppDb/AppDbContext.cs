using Microsoft.EntityFrameworkCore;

namespace SimpleTransformer.AppDb
{
    //For better future-proofing and being able to stop and resume training, along with register multiple jobs, I am going to create a database.
    public class AppDbContext : DbContext
    {
        
    }
}