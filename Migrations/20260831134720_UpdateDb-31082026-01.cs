using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace SimpleTransformer.Migrations
{
    /// <inheritdoc />
    public partial class UpdateDb3108202601 : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "InputFilePath",
                table: "TrainingJobs",
                type: "TEXT",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "InputText",
                table: "TrainingJobs",
                type: "TEXT",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "PreviousCheckpoint",
                table: "TrainingJobs",
                type: "TEXT",
                nullable: true);

            migrationBuilder.AddColumn<Guid>(
                name: "PreviousCheckpointId",
                table: "TrainingJobs",
                type: "TEXT",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "InputFilePath",
                table: "TrainingJobs");

            migrationBuilder.DropColumn(
                name: "InputText",
                table: "TrainingJobs");

            migrationBuilder.DropColumn(
                name: "PreviousCheckpoint",
                table: "TrainingJobs");

            migrationBuilder.DropColumn(
                name: "PreviousCheckpointId",
                table: "TrainingJobs");
        }
    }
}
