using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace SimpleTransformer.Migrations
{
    /// <inheritdoc />
    public partial class UpdateDb3008202601 : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateIndex(
                name: "IX_TransformerModels_Name",
                table: "TransformerModels",
                column: "Name",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_TransformerModels_TrainingConfigId",
                table: "TransformerModels",
                column: "TrainingConfigId");

            migrationBuilder.CreateIndex(
                name: "IX_TransformerModels_TransformerConfigId",
                table: "TransformerModels",
                column: "TransformerConfigId");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "IX_TransformerModels_Name",
                table: "TransformerModels");

            migrationBuilder.DropIndex(
                name: "IX_TransformerModels_TrainingConfigId",
                table: "TransformerModels");

            migrationBuilder.DropIndex(
                name: "IX_TransformerModels_TransformerConfigId",
                table: "TransformerModels");
        }
    }
}
