using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using SimpleTransformer.Config;

namespace SimpleTransformer.Api
{
    public static class DbContextConfiguration
    {
        public static void ConfigureDbContext(
            DbContextOptionsBuilder options, ConfigManager configurationManager)
        {
            Console.WriteLine("Configuring database...");
            var dbEngine = configurationManager
                .GetValueOrDefault("Database_Engine", "Sqlite", "General");

            switch (dbEngine.ToLowerInvariant())
            {
                case "sqlite":
                {
                    var dbname = configurationManager.GetValueOrDefault(
                        "Connection_String",
                        "data.db",
                        "Sqlite"  
                    );

                    // 1. Set connection string flags for timeout & pooling
                    var connectionStringBuilder = new SqliteConnectionStringBuilder
                    {
                        DataSource = dbname,
                        Mode = SqliteOpenMode.ReadWriteCreate,
                        Pooling = true,
                        DefaultTimeout = 5
                    };

                    options.UseSqlite(connectionStringBuilder.ConnectionString, sqliteOptions =>
                    {
                        sqliteOptions.CommandTimeout(30);
                    });

                    break;
                }

                case "sqlserver":
                {
                    var connectionString = configurationManager.GetValueOrDefault(
                        "Connection_String",
                        string.Empty,
                        "SqlServer");

                    options.UseSqlServer(connectionString);
                    break;
                }

                case "postgres":
                {
                    var connectionString = configurationManager.GetValueOrDefault(
                        "Connection_String",
                        string.Empty,
                        "Postgres");

                    options.UseNpgsql(connectionString);
                    break;
                }

                case "mysql":
                {
                    var connectionString = configurationManager.GetValueOrDefault(
                        "Connection_String",
                        string.Empty,
                        "MySql");

                    options.UseMySql(
                        connectionString,
                        ServerVersion.AutoDetect(connectionString));

                    break;
                }

                default:
                    throw new ArgumentException(
                        $"Unknown database engine: {dbEngine}");
            }
        }
        public static void InitializeDatabase(DbContext dbContext)
        {
            // Apply pending EF migrations automatically
            dbContext.Database.Migrate();

            // Run SQLite-specific PRAGMA tuning
            if (dbContext.Database.IsSqlite())
            {
                dbContext.Database.ExecuteSqlRaw("PRAGMA journal_mode = WAL;");
                dbContext.Database.ExecuteSqlRaw("PRAGMA synchronous = FULL;");
                dbContext.Database.ExecuteSqlRaw("PRAGMA foreign_keys = ON;");
            }
        }
    }
}